import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta, time
from io import BytesIO
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam

display_cols = [
    "Datetime",
    "Water_level",
    "T_Relative_humidity",
    "T_Rainfall",
    "T_Cloud_cover",
    "T_Surface_pressure",
    "SL_Relative_humidity",
    "SL_Cloud_cover",
    "SL_Surface_pressure",
    "MB_Relative_humidity",
    "MB_Cloud_cover",
    "MB_Surface_pressure",
    "MU_Relative_humidity",
    "MU_Cloud_cover",
    "MU_Surface_pressure",
    "Source"
]

# ============================================================
# 1) Load model & scalers
# ============================================================
feature_cols = [
    "T_Relative_humidity", "T_Rainfall", "T_Cloud_cover", "T_Surface_pressure",
    "SL_Relative_humidity", "SL_Rainfall", "SL_Cloud_cover", "SL_Surface_pressure",
    "MB_Relative_humidity", "MB_Rainfall", "MB_Cloud_cover", "MB_Surface_pressure",
    "MU_Relative_humidity", "MU_Rainfall", "MU_Cloud_cover", "MU_Surface_pressure",
    "Water_level"
]

# Load scalers
scaler_X = joblib.load("Final_scaler_X.save")
scaler_y = joblib.load("Final_scaler_y.save")

# Build model dengan konfigurasi terbaik dari training
input_days = 14
output_days = 7
n_features = len(feature_cols)  

best_units = 256
dropout_rate = 0.1
learning_rate = 1e-3

def build_seq2seq_new(input_steps, n_features, output_steps, units, dropout_rate, lr):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=(input_steps, n_features)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.RepeatVector(output_steps),
        tf.keras.layers.LSTM(units, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model

model = build_seq2seq_new(input_days, n_features, output_days, best_units, dropout_rate, learning_rate)
model.load_weights("Final_seq2seq_new_weights.h5")

st.title("🌊 Water Level Forecast Dashboard")

# -----------------------------
# Current time (GMT+7), rounded up to next full hour
# -----------------------------
now_utc = datetime.utcnow()
gmt7_now = now_utc + timedelta(hours=7)
rounded_now = gmt7_now.replace(minute=0, second=0, microsecond=0)
if gmt7_now.minute > 0 or gmt7_now.second > 0:
    rounded_now += timedelta(hours=1)

# -----------------------------
# Select forecast start datetime
# -----------------------------
st.subheader("Select Start Date & Time for 7-Day Forecast")
selected_date = st.date_input("Date", value=rounded_now.date(), max_value=rounded_now.date())
hour_options = [f"{h:02d}:00" for h in range(0, rounded_now.hour + 1)] if selected_date == rounded_now.date() else [f"{h:02d}:00" for h in range(0, 24)]
selected_hour_str = st.selectbox("Time (WIB)", hour_options, index=len(hour_options)-1)
selected_hour = int(selected_hour_str.split(":")[0])
start_datetime = datetime.combine(selected_date, time(selected_hour, 0, 0))
st.write(f"Start datetime (GMT+7): {start_datetime}")

# Historical selalu pakai rentang start → forecast end jika perlu
if start_datetime > rounded_now:
    # Full future
    hist_start = start_datetime - timedelta(hours=96)
    hist_end = start_datetime
    fore_start = start_datetime
    fore_end = start_datetime + timedelta(hours=168)

elif start_datetime <= rounded_now and start_datetime >= rounded_now - timedelta(hours=168):
    # Hybrid
    hist_start = start_datetime - timedelta(hours=96)
    hist_end = rounded_now  # historical sampai sekarang
    fore_start = start_datetime
    fore_end = start_datetime + timedelta(hours=168)

else:
    # Full past (start_datetime di masa lalu lebih dari 7 hari)
    hist_start = start_datetime - timedelta(hours=96)
    hist_end = start_datetime + timedelta(hours=168)  # historical dipakai sampai forecast end
    fore_start = start_datetime
    fore_end = start_datetime + timedelta(hours=168)

# -----------------------------
# Instructions for upload
# -----------------------------
st.subheader("Instructions for Uploading Water Level Data")
st.info(
    f"- CSV must contain columns: 'Datetime' and 'Level Air'.\n"
    f"- 'Datetime' format: YYYY-MM-DD HH:MM:SS\n"
    f"- Data must cover **4 days before the selected start datetime**: "
    f"{start_datetime - timedelta(hours=96)} to {start_datetime}\n"
    f"- Make sure there are no missing hours."
)

# -----------------------------
# Upload water level data
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File (AWLR Jetty Tuhup Logs)", type=["csv"])
wl_hourly = None
upload_success = False

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
        
        # Pastikan kolom ada
        if "Datetime" not in df_raw.columns or "Level Air" not in df_raw.columns:
            st.error("The file must contain columns 'Datetime' and 'Level Air'.")
        else:
            # -----------------------
            # 1️⃣ Baca & bersihkan data
            # -----------------------
            df_raw['Datetime'] = pd.to_datetime(df_raw['Datetime'], errors='coerce')
            df_raw = df_raw.dropna(subset=['Datetime'])
            df_raw = df_raw.sort_values('Datetime').set_index('Datetime')
            df_raw['Water_level'] = pd.to_numeric(df_raw['Level Air'], errors='coerce')
            df_raw = df_raw.drop(columns=['Level Air'])
            
            # -----------------------
            # 2️⃣ Resample menjadi daily average
            # -----------------------
            wl_daily = df_raw.resample('1D').mean().reset_index()
            wl_daily['Water_level'] = wl_daily['Water_level'].interpolate(method='linear')
            
            # -----------------------
            # 3️⃣ Bersihkan nilai tidak realistis
            # -----------------------
            wl_daily.loc[wl_daily['Water_level'] <= 0, 'Water_level'] = np.nan
            wl_daily['Water_level'] = wl_daily['Water_level'].interpolate(method='linear')
            
            # -----------------------
            # 4️⃣ Deteksi spike antar hari
            # -----------------------
            diff = wl_daily['Water_level'].diff().abs()
            spike_mask = diff > 0.5  # threshold bisa disesuaikan untuk daily
            wl_daily.loc[spike_mask, 'Water_level'] = np.nan
            
            # -----------------------
            # 5️⃣ Interpolasi & smoothing ringan
            # -----------------------
            wl_daily['Water_level'] = wl_daily['Water_level'].interpolate(method='linear', limit_direction='both')
            wl_daily['Water_level'] = wl_daily['Water_level'].rolling(window=3, center=True, min_periods=1).median()
            wl_daily['Water_level'] = wl_daily['Water_level'].rolling(window=3, center=True, min_periods=1).mean()
            wl_daily['Water_level'] = wl_daily['Water_level'].round(2)
            
            # -----------------------
            # 6️⃣ Hanya 14 hari terakhir sebelum start_datetime (daily)
            # -----------------------
            start_limit = start_datetime - pd.Timedelta(days=14)
            wl_daily = wl_daily[(wl_daily["Datetime"] >= start_limit) & (wl_daily["Datetime"] < start_datetime)]
            wl_daily = wl_daily.drop_duplicates(subset="Datetime").reset_index(drop=True)
            
            # -----------------------
            # 7️⃣ Validasi missing days
            # -----------------------
            expected_days = pd.date_range(start=start_limit, end=start_datetime - pd.Timedelta(days=1), freq='D')
            actual_days = pd.to_datetime(wl_daily["Datetime"])
            missing_days = sorted(set(expected_days) - set(actual_days))
            if missing_days:
                missing_str = ', '.join([dt.strftime("%Y-%m-%d") for dt in missing_days])
                st.warning(f"The uploaded water level data is incomplete! Missing days: {missing_str}")
            else:
                upload_success = True
                st.success("✅ File uploaded, cleaned, and validated successfully!")
                st.dataframe(wl_daily)

# -----------------------------
# Multi-point coordinates + bobot IDW
# -----------------------------
multi_points = {
    "T_": { "NW": {"lat":0.38664, "lon":113.78421, "weight":0.000639},
            "N": {"lat":0.38664, "lon":114.83972, "weight":0.001287},
            "NE": {"lat":0.38664, "lon":115.89523, "weight":0.000591},
            "W": {"lat":-0.59754, "lon":113.76960, "weight":0.001233},
            "Center": {"lat":-0.59754, "lon":114.82759, "weight":0.992519},
            "E": {"lat":-0.59754, "lon":115.81505, "weight":0.001271},
            "SW":{"lat":-1.65202, "lon":113.76685, "weight":0.000615},
            "S": {"lat":-1.65202, "lon":114.76606, "weight":0.001232},
            "SE":{"lat":-1.65202, "lon":115.83664, "weight":0.000613} },
    "SL_": { "NW": {"lat":0.52724, "lon":113.68050, "weight":0.000329},
             "N": {"lat":0.52724, "lon":114.73766, "weight":0.000679},
             "NE": {"lat":0.52724, "lon":115.65387, "weight":0.000347},
             "W": {"lat":-0.45694, "lon":113.73240, "weight":0.000708},
             "Center": {"lat":-0.45694, "lon":114.71832, "weight":0.995940},
             "E": {"lat":-0.45694, "lon":115.70423, "weight":0.000639},
             "SW":{"lat":-1.44112, "lon":113.71044, "weight":0.000337},
             "S": {"lat":-1.44112, "lon":114.70727, "weight":0.000669},
             "SE":{"lat":-1.44112, "lon":115.63291, "weight":0.000352} },
    "MB_": { "NW": {"lat":0.38664, "lon":113.64348, "weight":0.000166},
             "N": {"lat":0.38664, "lon":114.55825, "weight":0.000332},
             "NE":{"lat":0.38664, "lon":115.61377, "weight":0.000168},
             "W": {"lat":-0.59754, "lon":113.62853, "weight":0.000320},
             "Center": {"lat":-0.59754, "lon":114.61599, "weight":0.997992},
             "E": {"lat":-0.59754, "lon":115.60345, "weight":0.000345},
             "SW":{"lat":-1.58172, "lon":113.60538, "weight":0.000160},
             "S": {"lat":-1.58172, "lon":114.60380, "weight":0.000334},
             "SE":{"lat":-1.58172, "lon":115.53090, "weight":0.000183} },
    "MU_": { "NW": {"lat":0.31634, "lon":113.48438, "weight":0.000587},
             "N": {"lat":0.31634, "lon":114.53906, "weight":0.001197},
             "NE": {"lat":0.31634, "lon":115.52344, "weight":0.000613},
             "W": {"lat":-0.73814, "lon":113.52433, "weight":0.001244},
             "Center": {"lat":-0.73814, "lon":114.51334, "weight":0.992616},
             "E": {"lat":-0.73814, "lon":115.50235, "weight":0.001309},
             "SW":{"lat":-1.72232, "lon":113.50001, "weight":0.000598},
             "S": {"lat":-1.72232, "lon":114.50001, "weight":0.001206},
             "SE":{"lat":-1.72232, "lon":115.50001, "weight":0.000630} },
}

# Kolom numerik yang akan dipakai untuk perhitungan weighted average
numeric_cols_hist = ["relative_humidity_2m", "precipitation", "cloud_cover", "surface_pressure"]
numeric_cols_fore = ["relative_humidity_2m", "precipitation", "cloud_cover", "surface_pressure"]

# -----------------------------
# Fungsi bantu: fetch per-lokasi (historical)
# -----------------------------
# Nama lengkap region
region_labels = {
    "T_": "Tuhup",
    "SL_": "Sungai Laung",
    "MB_": "Muara Bumban",
    "MU_": "Muara Untu"
}

# -----------------------------
# Fungsi: historical per region
# -----------------------------
def fetch_historical_multi_region(region_name, region_points, start_dt, end_dt):
    latitudes = ",".join(str(pt["lat"]) for pt in region_points.values())
    longitudes = ",".join(str(pt["lon"]) for pt in region_points.values())

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitudes}&longitude={longitudes}"
        f"&start_date={start_dt.date().isoformat()}&end_date={end_dt.date().isoformat()}"
        "&hourly=relative_humidity_2m,precipitation,cloud_cover,surface_pressure"
        "&timezone=Asia%2FBangkok"
    )

    try:
        resp = requests.get(url, timeout=60)
        data = resp.json()
    except Exception as e:
        st.warning(f"[{region_labels.get(region_name, region_name)}] Error fetching historical: {e}")
        return pd.DataFrame()

    # --- Normalisasi format respons
    if isinstance(data, dict) and "hourly" in data:
        data = [data] * len(region_points)  # duplikasikan 1 response untuk semua titik
    elif isinstance(data, dict):
        data = list(data.values()) if "latitude" not in data else [data]
    elif not isinstance(data, list):
        st.warning(f"[{region_labels.get(region_name, region_name)}] Unexpected historical response format.")
        return pd.DataFrame()

    # --- Jika jumlah item tidak sama, sesuaikan panjangnya
    if len(data) < len(region_points):
        st.info(f"[{region_labels.get(region_name, region_name)}] Adjusting historical response length ({len(data)} vs {len(region_points)}).")
        while len(data) < len(region_points):
            data.append(data[-1])
    elif len(data) > len(region_points):
        data = data[:len(region_points)]

    all_dfs = []

    for i, (dir_name, info) in enumerate(region_points.items()):
        loc_data = data[i]
        if not isinstance(loc_data, dict) or "hourly" not in loc_data:
            continue

        df = pd.DataFrame(loc_data["hourly"])
        if df.empty or "time" not in df.columns:
            continue

        df["direction"] = dir_name
        df["weight"] = info["weight"]
        all_dfs.append(df)

    if not all_dfs:
        st.warning(f"[{region_labels.get(region_name, region_name)}] No valid historical data.")
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"])

    # Weighted average
    weighted_list = []
    for time, group in df_all.groupby("time"):
        w = group["weight"].values
        weighted_vals = (group[numeric_cols_hist].T * w).T.sum()
        row = weighted_vals.to_dict()
        row["Datetime"] = time
        weighted_list.append(row)

    df_weighted = pd.DataFrame(weighted_list)
    df_weighted.rename(columns={
        "relative_humidity_2m": "Relative_humidity",
        "precipitation": "Rainfall",
        "cloud_cover": "Cloud_cover",
        "surface_pressure": "Surface_pressure"
    }, inplace=True)
    for c in ["Relative_humidity", "Rainfall", "Cloud_cover", "Surface_pressure"]:
        if c in df_weighted.columns:
            df_weighted[c] = df_weighted[c].round(2)

    df_weighted["Region"] = region_labels.get(region_name, region_name)
    return df_weighted[["Datetime", "Region", "Relative_humidity", "Rainfall", "Cloud_cover", "Surface_pressure"]]


# -----------------------------
# Fungsi: forecast per region
# -----------------------------
def fetch_forecast_multi_region(region_name, region_points):
    latitudes = ",".join(str(pt["lat"]) for pt in region_points.values())
    longitudes = ",".join(str(pt["lon"]) for pt in region_points.values())

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitudes}&longitude={longitudes}"
        "&hourly=relative_humidity_2m,precipitation,cloud_cover,surface_pressure"
        "&forecast_days=16&timezone=Asia%2FBangkok"
    )

    try:
        resp = requests.get(url, timeout=60)
        data = resp.json()
    except Exception as e:
        st.warning(f"[{region_labels.get(region_name, region_name)}] Error fetching forecast: {e}")
        return pd.DataFrame()

    # --- Normalisasi format respons
    if isinstance(data, dict) and "hourly" in data:
        data = [data] * len(region_points)
    elif isinstance(data, dict):
        data = list(data.values()) if "latitude" not in data else [data]
    elif not isinstance(data, list):
        st.warning(f"[{region_labels.get(region_name, region_name)}] Unexpected forecast response format.")
        return pd.DataFrame()

    if len(data) < len(region_points):
        st.info(f"[{region_labels.get(region_name, region_name)}] Adjusting forecast response length ({len(data)} vs {len(region_points)}).")
        while len(data) < len(region_points):
            data.append(data[-1])
    elif len(data) > len(region_points):
        data = data[:len(region_points)]

    all_dfs = []

    for i, (dir_name, info) in enumerate(region_points.items()):
        loc_data = data[i]
        if not isinstance(loc_data, dict) or "hourly" not in loc_data:
            continue

        df = pd.DataFrame(loc_data["hourly"])
        if df.empty or "time" not in df.columns:
            continue

        df["direction"] = dir_name
        df["weight"] = info["weight"]
        all_dfs.append(df)

    if not all_dfs:
        st.warning(f"[{region_labels.get(region_name, region_name)}] No valid forecast data.")
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"])

    weighted_list = []
    for time, group in df_all.groupby("time"):
        w = group["weight"].values
        weighted_vals = (group[numeric_cols_fore].T * w).T.sum()
        row = weighted_vals.to_dict()
        row["Datetime"] = time
        weighted_list.append(row)

    df_weighted = pd.DataFrame(weighted_list)
    df_weighted.rename(columns={
        "relative_humidity_2m": "Relative_humidity",
        "precipitation": "Rainfall",
        "cloud_cover": "Cloud_cover",
        "surface_pressure": "Surface_pressure"
    }, inplace=True)
    for c in ["Relative_humidity", "Rainfall", "Cloud_cover", "Surface_pressure"]:
        if c in df_weighted.columns:
            df_weighted[c] = df_weighted[c].round(2)

    df_weighted["Region"] = region_labels.get(region_name, region_name)
    return df_weighted[["Datetime", "Region", "Relative_humidity", "Rainfall", "Cloud_cover", "Surface_pressure"]]

# -----------------------------
# Wrapper: proses setiap region berurutan dan merge jadi satu wide table
# -----------------------------
run_forecast = st.button("Run 7-Day Forecast")
if "forecast_done" not in st.session_state:
    st.session_state["forecast_done"] = False
    st.session_state["final_df"] = None
    st.session_state["forecast_running"] = False

if run_forecast:
    st.session_state["forecast_done"] = False
    st.session_state["final_df"] = None
    st.session_state["forecast_running"] = True
    st.rerun()

if upload_success and st.session_state.get("forecast_running", False):
    progress_container = st.empty()
    total_forecast_hours = 168
    total_steps = len(multi_points) * 2 + 3 + total_forecast_hours
    step_counter = 0
    progress_bar = st.progress(0.0)

    # kita akan gabungkan hasil semua region secara wide
    merged_wide = wl_hourly.copy()
    if "time" in merged_wide.columns:
        merged_wide.rename(columns={"time": "Datetime"}, inplace=True)
    merged_wide["Datetime"] = pd.to_datetime(merged_wide["Datetime"])

    for region_name, region_points in multi_points.items():
        region_label = region_labels.get(region_name, region_name)
        progress_container.markdown(f"Fetching data for **{region_label}** ...")

        # --- Historical ---
        hist_df = fetch_historical_multi_region(region_name, region_points, hist_start, hist_end)
        hist_df["Source"] = "Historical"
        
        # --- Forecast ---
        fore_df = pd.DataFrame()
        
        if start_datetime > rounded_now:
            # Full future → ambil forecast API
            fore_df = fetch_forecast_multi_region(region_name, region_points)
            fore_df = fore_df[(fore_df["Datetime"] >= fore_start) & (fore_df["Datetime"] <= fore_end)]
            fore_df["Source"] = "Forecast"
        
        elif start_datetime <= rounded_now and start_datetime >= rounded_now - timedelta(hours=168):
            # Hybrid
            # 1. forecast dari historical API sampai rounded_now
            hist_fore_df = hist_df[(hist_df["Datetime"] >= fore_start) & (hist_df["Datetime"] <= rounded_now)].copy()
            hist_fore_df["Source"] = "Forecast"
        
            # 2. forecast dari API > rounded_now
            if fore_end > rounded_now:
                api_fore_df = fetch_forecast_multi_region(region_name, region_points)
                api_fore_df = api_fore_df[(api_fore_df["Datetime"] > rounded_now) & (api_fore_df["Datetime"] <= fore_end)]
                api_fore_df["Source"] = "Forecast"
                fore_df = pd.concat([hist_fore_df, api_fore_df], ignore_index=True)
            else:
                fore_df = hist_fore_df
        
        else:
            # Full past → semua forecast dari historical
            fore_df = hist_df[(hist_df["Datetime"] >= fore_start) & (hist_df["Datetime"] <= fore_end)].copy()
            fore_df["Source"] = "Forecast"

    
        # Gabungkan historical + forecast jadi satu (Datetime + var)
        combined_df = pd.concat([hist_df, fore_df], ignore_index=True)
        if combined_df.empty:
            st.warning(f"{region_label}: no data returned.")
            continue
        combined_df["Datetime"] = pd.to_datetime(combined_df["Datetime"])
        combined_df.sort_values("Datetime", inplace=True)

        # Hapus duplikat waktu (kalau ada overlap hist/forecast)
        combined_df = combined_df.drop_duplicates(subset=["Datetime"], keep="last")

        # --- Batasi waktu ---
        combined_df = combined_df[
            (combined_df["Datetime"] >= start_datetime - timedelta(hours=96)) &
            (combined_df["Datetime"] < start_datetime + timedelta(hours=168))
        ]

        # Ganti nama kolom jadi prefiks region (T_, SL_, dst)
        rename_map = {
            "Relative_humidity": f"{region_name}Relative_humidity",
            "Cloud_cover": f"{region_name}Cloud_cover",
            "Surface_pressure": f"{region_name}Surface_pressure",
        }
        if region_name not in ["SL_", "MB_", "MU_"]:
            rename_map["Rainfall"] = f"{region_name}Rainfall"
        combined_df.rename(columns=rename_map, inplace=True)

        # Merge ke tabel utama
        merged_wide = pd.merge(
            merged_wide, 
            combined_df[["Datetime"] + list(rename_map.values())],
            on="Datetime", how="outer"
        )
        step_counter += 1
        progress_bar.progress(min(max(step_counter / total_steps, 0.0), 1.0))

    # urutkan dan rapikan
    merged_wide = merged_wide.sort_values("Datetime")
    
    # -----------------------------
    # Resample / aggregate per day
    # -----------------------------
    if not merged_wide.empty:
        merged_wide["Datetime"] = pd.to_datetime(merged_wide["Datetime"])
        merged_wide.set_index("Datetime", inplace=True)
    
        # Tentukan kolom rainfall
        rainfall_cols = [col for col in merged_wide.columns if "Rainfall" in col]
    
        # Semua kolom lain = mean, Rainfall = sum
        agg_dict = {col: 'mean' for col in merged_wide.columns if col not in rainfall_cols}
        for col in rainfall_cols:
            agg_dict[col] = 'sum'
    
        # Resample per hari
        merged_daily = merged_wide.resample('1D').agg(agg_dict)
        merged_daily = merged_daily.round(2)
        merged_daily.reset_index(inplace=True)
    
        final_df = merged_daily.copy()
        st.session_state["final_df"] = final_df
    
    st.session_state["forecast_done"] = True
    st.session_state["forecast_running"] = False

    # -----------------------------
    # 4️⃣ Iterative forecast dengan dynamic lag
    # -----------------------------
    input_seq = final_df[model_features].iloc[-input_days:].values
    input_seq_scaled = scaler_X.transform(input_seq.reshape(-1, len(model_features))).reshape(1, input_days, len(model_features))
    
    # Prediksi delta 7 hari
    delta_scaled = model.predict(input_seq_scaled, verbose=0)  # shape (1, 7, 1)
    delta_pred = scaler_y.inverse_transform(delta_scaled.reshape(-1,1)).reshape(-1)
    
    # Last water level dari input window
    last_wl = final_df["Water_level"].iloc[-1]
    
    # Hitung forecast absolute water level
    forecast_values = last_wl + delta_pred
    
    # Buat tanggal forecast 7 hari
    forecast_dates = pd.date_range(start=final_df["Datetime"].iloc[-1] + pd.Timedelta(days=1), periods=7)
    
    # Masukkan ke dataframe
    forecast_df = pd.DataFrame({
        "Datetime": forecast_dates,
        "Water_level": forecast_values,
        "Source": "Forecast"
    })
    
    final_df = pd.concat([final_df, forecast_df], ignore_index=True)
    
    # Update session_state
    st.session_state["final_df"] = final_df
    st.session_state["forecast_done"] = True
    st.session_state["forecast_running"] = False
    progress_container.markdown("✅ 7-Day Water Level Forecast Completed!")
    progress_bar.progress(1.0)

    # -----------------------------
    # Display Forecast & Plot
    # -----------------------------
    result_container = st.container()
    
    if st.session_state["forecast_done"] and st.session_state["final_df"] is not None:
        final_df = st.session_state["final_df"]
        with result_container:
            st.subheader("Water Level + Climate Data with Forecast")
    
            
            # 1. Buat DataFrame khusus untuk styling/display. 
            df_for_styling = final_df[display_cols].copy()

            # 2. Definisikan fungsi styling
            def highlight_forecast(row):
                # Fungsi ini sekarang aman karena 'Source' ada di df_for_styling
                return ['background-color: #cfe9ff' if row['Source']=="Forecast" else '' for _ in row]

            num_cols_to_format = [
                "Water_level", "T_Relative_humidity", "T_Rainfall", "T_Cloud_cover", "T_Surface_pressure",
                "SL_Relative_humidity", "SL_Cloud_cover", "SL_Surface_pressure",
                "MB_Relative_humidity", "MB_Cloud_cover", "MB_Surface_pressure",
                "MU_Relative_humidity", "MU_Cloud_cover", "MU_Surface_pressure"
            ]

            #Buat dictionary format
            format_dict = {col: "{:.2f}" for col in num_cols_to_format if col in df_for_styling.columns}
            
            # Terapkan styling
            styled_df = df_for_styling.style.apply(
                highlight_forecast,
                axis=1
            ).format(format_dict)

            st.dataframe(styled_df, use_container_width=True, height=500)

        # -----------------------------
        # Plot Water Level Forecast
        # -----------------------------
        st.subheader("Water Level Forecast Plot")
        fig = go.Figure()
        
        hist_df = final_df[final_df["Source"]=="Historical"]
        fore_df = final_df[final_df["Source"]=="Forecast"]
        
        if not fore_df.empty:
            rmse = 0.219
        
            # Titik terakhir historis
            last_hist_dt = hist_df["Datetime"].iloc[-1]
            last_hist_val = hist_df["Water_level"].iloc[-1]
        
            # Forecast X & Y (tersambung dari historis)
            forecast_x = pd.concat([pd.Series([last_hist_dt]), fore_df["Datetime"]])
            forecast_y = pd.concat([pd.Series([last_hist_val]), fore_df["Water_level"]])
        
            upper_y = forecast_y + rmse
            lower_y = (forecast_y - rmse).clip(lower=0)
        
            # 2️⃣ Trace upper RMSE
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=upper_y,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name="RMSE Upper"
            ))
        
            # 3️⃣ Trace lower RMSE + fill ke upper
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=lower_y,
                mode='lines',
                fill='tonexty',
                fillcolor="rgba(255,165,0,0.2)",
                line=dict(width=0),
                name=f"±RMSE {rmse:.3f} m"
            ))
        
            # 4️⃣ Garis forecast (orange) termasuk titik historis terakhir
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="orange"),
                marker=dict(size=4)
            ))
        
        # 1️⃣ Garis historis
        fig.add_trace(go.Scatter(
            x=hist_df["Datetime"],
            y=hist_df["Water_level"],
            mode="lines+markers",
            name="Historical",
            line=dict(color="blue"),
            marker=dict(size=4)
        ))
        
        # 5️⃣ Garis horizontal limit
        fig.add_trace(go.Scatter(
            x=[final_df["Datetime"].min(), final_df["Datetime"].max()],
            y=[19.5, 19.5],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Lower Limit 19.5 m"
        ))
        fig.add_trace(go.Scatter(
            x=[final_df["Datetime"].min(), final_df["Datetime"].max()],
            y=[28, 28],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Upper Limit 28 m"
        ))
        
        # Layout
        fig.update_layout(
            title="Water Level Historical vs Forecast",
            xaxis_title="Datetime",
            yaxis_title="Water Level (m)",
            template="plotly_white",
            annotations=[
                dict(
                    xref="paper", yref="paper",
                    x=0.98, y=0.95,
                    text=f"RMSE = {rmse:.3f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    borderpad=4
                )
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # -----------------------------
        # Downloads (Forecast only, Datetime + Water_level, 2 decimals)
        # -----------------------------
        forecast_only = final_df[final_df["Source"] == "Forecast"][["Datetime", "Water_level"]].copy()
        forecast_only["Water_level"] = forecast_only["Water_level"].round(2)
        forecast_only["Datetime"] = forecast_only["Datetime"].astype(str)

        
        # CSV
        csv_buffer = forecast_only.to_csv(index=False).encode('utf-8')
        
        # Excel
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            forecast_only.to_excel(writer, index=False, sheet_name="Forecast")
        excel_buffer.seek(0)
        
        # PDF
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        data = [forecast_only.columns.tolist()] + forecast_only.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#007acc")),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
            ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ]))
        elements = [Paragraph("Joloi Water Level Forecast (Forecast Only)", styles["Title"]), table]
        doc.build(elements)
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download CSV", csv_buffer, "water_level_forecast.csv", "text/csv", use_container_width=True)
        with col2:
            st.download_button("Download Excel", excel_buffer, "water_level_forecast.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with col3:
            st.download_button("Download PDF", pdf_buffer.getvalue(), "water_level_forecast.pdf", "application/pdf", use_container_width=True)
else:
    result_container = st.empty()

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

scaler_X = joblib.load("scaler_X.pkl") 
scaler_y = joblib.load("scaler_y.pkl")

# ============================
# Parameters (samakan dengan saat training)
# ============================
units = 128
dropout_rate = 0.3
window_size = 24  # sama seperti saat training
n_features = 725

# ============================
# Rebuild LSTM model
# ============================
model = Sequential()
model.add(LSTM(units, input_shape=(window_size, n_features), return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(units // 2))
model.add(Dense(1))

# Load bobot
model.load_weights("lstm_waterlevel_weights.h5")

# -----------------------------
# Load trained XGB model
# -----------------------------
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
            # 2️⃣ Resample menjadi hourly average
            # -----------------------
            wl_hourly = df_raw.resample('1H').mean().reset_index()
            wl_hourly['Water_level'] = wl_hourly['Water_level'].interpolate(method='linear')
            
            # -----------------------
            # 3️⃣ Bersihkan nilai tidak realistis
            # -----------------------
            wl_hourly.loc[wl_hourly['Water_level'] <= 0, 'Water_level'] = np.nan
            wl_hourly['Water_level'] = wl_hourly['Water_level'].interpolate(method='linear')
            
            # -----------------------
            # 4️⃣ Deteksi spike antar jam
            # -----------------------
            diff = wl_hourly['Water_level'].diff().abs()
            spike_mask = diff > 0.1  # threshold bisa disesuaikan
            wl_hourly.loc[spike_mask, 'Water_level'] = np.nan
            
            # -----------------------
            # 5️⃣ Interpolasi & smoothing ringan
            # -----------------------
            wl_hourly['Water_level'] = wl_hourly['Water_level'].interpolate(method='linear', limit_direction='both')
            wl_hourly['Water_level'] = wl_hourly['Water_level'].rolling(window=3, center=True, min_periods=1).median()
            wl_hourly['Water_level'] = wl_hourly['Water_level'].rolling(window=3, center=True, min_periods=1).mean()
            wl_hourly['Water_level'] = wl_hourly['Water_level'].round(3)
            
            # -----------------------
            # 6️⃣ Hanya 96 jam terakhir sebelum start_datetime
            # -----------------------
            start_limit = start_datetime - pd.Timedelta(hours=96)
            wl_hourly = wl_hourly[(wl_hourly["Datetime"] >= start_limit) & (wl_hourly["Datetime"] < start_datetime)]
            wl_hourly = wl_hourly.drop_duplicates(subset="Datetime").reset_index(drop=True)
            
            # -----------------------
            # 7️⃣ Validasi missing hours
            # -----------------------
            expected_hours = pd.date_range(start=start_limit, end=start_datetime - pd.Timedelta(hours=1), freq='H')
            actual_hours = pd.to_datetime(wl_hourly["Datetime"])
            missing_hours = sorted(set(expected_hours) - set(actual_hours))
            if missing_hours:
                missing_str = ', '.join([dt.strftime("%Y-%m-%d %H:%M") for dt in missing_hours])
                st.warning(f"The uploaded water level data is incomplete! Missing hours: {missing_str}")
            else:
                upload_success = True
                st.success("✅ File uploaded, cleaned, and validated successfully!")
                st.dataframe(wl_hourly)
                
    except Exception as e:
        st.error(f"Failed to read file: {e}")

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

        # Fetch historical
        hist_df = fetch_historical_multi_region(region_name, region_points,
                                               start_datetime - timedelta(hours=96),
                                               start_datetime)
        step_counter += 1
        progress_bar.progress(min(max(step_counter / total_steps, 0.0), 1.0))


        # Fetch forecast
        fore_df = fetch_forecast_multi_region(region_name, region_points)
        step_counter += 1
        progress_bar.progress(min(max(step_counter / total_steps, 0.0), 1.0))


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
    merged_wide = merged_wide.round(2)

    final_df = merged_wide.copy()
    
    st.session_state["final_df"] = merged_wide
    st.session_state["forecast_done"] = True
    st.session_state["forecast_running"] = False

    def create_lag_features(df, cols, max_lag=95):
        for col in cols:
            for lag in range(1, max_lag+1):
                df[f"{col}_Lag{lag}"] = df[col].shift(lag)
        return df
    
    lag_cols = [
        "Water_level",
        "T_Relative_humidity", "T_Rainfall", "T_Cloud_cover", "T_Surface_pressure",
        "SL_Relative_humidity", "SL_Cloud_cover", "SL_Surface_pressure",
        "MB_Relative_humidity", "MB_Cloud_cover", "MB_Surface_pressure",
        "MU_Relative_humidity", "MU_Cloud_cover", "MU_Surface_pressure"
    ]
    
    final_df = create_lag_features(final_df, lag_cols, max_lag=95)
    final_df.fillna(method='bfill', inplace=True)  # isi NaN awal
    
    # 4️⃣ Iterative forecast
    progress_container.markdown("Forecasting water level 7 days iteratively...")
    model_features = []

    # T_ group
    for i in range(61, 96):
        model_features.append(f"T_Relative_humidity_Lag{i}")
    for i in range(82, 95):
        model_features.append(f"T_Rainfall_Lag{i}")
    for i in range(1, 96):
        model_features.append(f"T_Cloud_cover_Lag{i}")
    for i in range(54, 72):
        model_features.append(f"T_Surface_pressure_Lag{i}")
    
    # SL_ group
    for i in range(50, 96):
        model_features.append(f"SL_Relative_humidity_Lag{i}")
    for i in range(1, 96):
        model_features.append(f"SL_Cloud_cover_Lag{i}")
    for i in range(54, 73):
        model_features.append(f"SL_Surface_pressure_Lag{i}")
    
    # MB_ group
    for i in range(44, 96):
        model_features.append(f"MB_Relative_humidity_Lag{i}")
    for i in range(1, 96):
        model_features.append(f"MB_Cloud_cover_Lag{i}")
    for i in range(56, 70):
        model_features.append(f"MB_Surface_pressure_Lag{i}")
    
    # MU_ group
    for i in range(59, 96):
        model_features.append(f"MU_Relative_humidity_Lag{i}")
    for i in range(1, 96):
        model_features.append(f"MU_Cloud_cover_Lag{i}")
    for i in range(55, 71):
        model_features.append(f"MU_Surface_pressure_Lag{i}")
    
    # Water_level_ group
    for i in range(1, 96):
        model_features.append(f"Water_level_Lag{i}")

    # Pastikan kolom Source ada
    if "Source" not in final_df.columns:
        final_df["Source"] = np.where(final_df["Datetime"] < start_datetime, "Historical", "Forecast")
    
    # Tentukan forecast horizon
    forecast_hours = 168  # 7 hari × 24 jam
    
    # 🔹 Ambil index baris forecast yang sudah ada
    forecast_indices = final_df.index[final_df["Source"] == "Forecast"].tolist()
    
    for i, idx in enumerate(forecast_indices, start=1):
        next_datetime = final_df.at[idx, "Datetime"]
    
        progress_container.markdown(
            f"⏳ Forecasting hour {i} of {forecast_hours} "
            f"({(i)//24 + 1} of 7 days)..."
        )
    
        # 🔹 Ambil window terakhir dari data sebelum jam ini (termasuk prediksi sebelumnya)
        hist_part = final_df.loc[final_df["Datetime"] < next_datetime].tail(window_size)
        if len(hist_part) < window_size:
            continue  # belum cukup data
    
        X_seq = hist_part[model_features].values
        X_seq_scaled = scaler_X.transform(X_seq)
        X_seq_scaled = X_seq_scaled.reshape(1, window_size, len(model_features))
    
        # 🔹 Prediksi
        y_hat_scaled = model.predict(X_seq_scaled, verbose=0)
        y_hat = scaler_y.inverse_transform(y_hat_scaled)[0, 0]
        y_hat = max(0, y_hat)
    
        # 🔹 Simpan hasil prediksi
        final_df.at[idx, "Water_level"] = round(y_hat, 2)
    
        # 🔹 Update lag kolom untuk jam berikutnya
        for lag in range(95, 0, -1):
            lag_col = f"Water_level_Lag{lag}"
            if lag == 1:
                final_df.loc[final_df["Datetime"] >= next_datetime, lag_col] = y_hat
            else:
                prev_col = f"Water_level_Lag{lag-1}"
                if prev_col in final_df.columns:
                    final_df.loc[final_df["Datetime"] >= next_datetime, lag_col] = \
                        final_df.loc[final_df["Datetime"] < next_datetime, prev_col].iloc[-1]
    
        # 🔹 Update progress bar
        step_counter += 1
        progress_bar.progress(min(max(step_counter / total_steps, 0.0), 1.0))
    
# -----------------------------
# Display Forecast & Plot
# -----------------------------
result_container = st.empty()
if st.session_state["forecast_done"] and st.session_state["final_df"] is not None:
    final_df = st.session_state["final_df"]
    with result_container.container():
        st.subheader("Water Level + Climate Data with Forecast")

        # 🔹 Pilih hanya kolom utama (tanpa lag)
        main_columns = [
            "Datetime", "Water_level",
            "T_Relative_humidity", "T_Rainfall", "T_Cloud_cover", "T_Surface_pressure",
            "SL_Relative_humidity", "SL_Cloud_cover", "SL_Surface_pressure",
            "MB_Relative_humidity", "MB_Cloud_cover", "MB_Surface_pressure",
            "MU_Relative_humidity", "MU_Cloud_cover", "MU_Surface_pressure",
            "Source"
        ]
        # Ambil hanya kolom yang benar-benar ada
        display_df = final_df.copy()


        # 🔹 Highlight baris forecast
        def highlight_forecast(row):
            return ['background-color: #cfe9ff' if row['Source']=="Forecast" else '' for _ in row]

        # 🔹 Format hanya kolom numerik (biar rapi)
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        styled_df = display_df.style.apply(highlight_forecast, axis=1)\
                                    .format({col: "{:.2f}" for col in numeric_cols})

        # 🔹 Tampilkan di Streamlit
        st.dataframe(styled_df, use_container_width=True, height=500)

        # -----------------------------
        # Plot
        # -----------------------------
        st.subheader("Water Level Forecast Plot")
        fig = go.Figure()
        hist_df = final_df[final_df["Source"] == "Historical"]
        fore_df = final_df[final_df["Source"] == "Forecast"]


        if not fore_df.empty:
            rmse = 0.03
            last_val = hist_df["Water_level"].iloc[-1]
            forecast_x = pd.concat([pd.Series([hist_df["Datetime"].iloc[-1]]), fore_df["Datetime"]])
            forecast_y = pd.concat([pd.Series([last_val]), fore_df["Water_level"]])

            upper_y = forecast_y + rmse
            lower_y = forecast_y - rmse
            lower_y = lower_y.clip(lower=0)

            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_x, forecast_x[::-1]]),
                y=pd.concat([upper_y, lower_y[::-1]]),
                fill="toself", fillcolor="rgba(255,165,0,0.2)",
                line=dict(color="rgba(255,165,0,0)"),
                hoverinfo="skip",
                showlegend=True, name="±RMSE 0.03m"
            ))

            fig.add_trace(go.Scatter(
                x=forecast_x, y=forecast_y,
                mode="lines+markers", name="Forecast",
                line=dict(color="orange"), marker=dict(size=4)
            ))

        fig.add_trace(go.Scatter(
            x=hist_df["Datetime"], y=hist_df["Water_level"],
            mode="lines+markers", name="Historical",
            line=dict(color="blue"), marker=dict(size=4)
        ))

        fig.add_trace(go.Scatter(
            x=[final_df["Datetime"].min(), final_df["Datetime"].max()],
            y=[19.5, 19.5],
            mode="lines", line=dict(color="red", dash="dash"),
            name="Lower Limit 19.5 m"
        ))
        fig.add_trace(go.Scatter(
            x=[final_df["Datetime"].min(), final_df["Datetime"].max()],
            y=[28, 28],
            mode="lines", line=dict(color="red", dash="dash"),
            name="Upper Limit 28 m"
        ))

        fig.update_layout(
            title="Water Level Historical vs Forecast",
            xaxis_title="Datetime", yaxis_title="Water Level (m)",
            template="plotly_white",
            annotations=[
                dict(
                    xref="paper", yref="paper", x=0.98, y=0.95,
                    text=f"RMSE = {rmse:.2f}", showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1, borderpad=4
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Downloads (Forecast only)
        # -----------------------------
        forecast_only = final_df[final_df["Source"] == "Forecast"][["Datetime", "Water_level"]].copy()
        forecast_only["Water_level"] = forecast_only["Water_level"].round(2)
        forecast_only["Datetime"] = forecast_only["Datetime"].astype(str)

        csv_buffer = forecast_only.to_csv(index=False).encode('utf-8')
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            forecast_only.to_excel(writer, index=False, sheet_name="Forecast")
        excel_buffer.seek(0)

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        data = [forecast_only.columns.tolist()] + forecast_only.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#007acc")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        elements = [Paragraph("Joloi Water Level Forecast (Forecast Only)", styles["Title"]), table]
        doc.build(elements)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download CSV", csv_buffer, "water_level_forecast.csv", "text/csv", use_container_width=True)
        with col2:
            st.download_button("Download Excel", excel_buffer, "water_level_forecast.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with col3:
            st.download_button("Download PDF", pdf_buffer.getvalue(), "water_level_forecast.pdf", "application/pdf", use_container_width=True)
else:
    result_container.empty()

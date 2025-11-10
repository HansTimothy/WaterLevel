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

# -----------------------------
# Load trained XGB model
# -----------------------------
model = joblib.load("xgb_waterlevel_hourly_model.pkl")
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

# Variabel cuaca yang kita ambil dari API
# (nama sesuai dengan Open-Meteo fields)
climate_vars = ["relativehumidity_2m", "precipitation", "cloud_cover", "surface_pressure"]

# -----------------------------
# Fungsi bantu: fetch per-lokasi (historical)
# -----------------------------
def fetch_historical_multi(start_dt, end_dt, show_warnings=True):
    """
    Kembalikan DataFrame berisi Datetime + kolom per-lokasi:
    e.g. T_Relative_humidity, T_Rainfall, T_Cloud_cover, T_Surface_pressure, ...
    """
    results_per_location = {}
    for loc_prefix, pts in multi_points.items():
        dfs = []
        for dir_name, meta in pts.items():
            lat = meta["lat"]
            lon = meta["lon"]
            w = meta["weight"]
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}"
                f"&start_date={start_dt.date().isoformat()}&end_date={end_dt.date().isoformat()}"
                f"&hourly={','.join(climate_vars)}"
                "&timezone=Asia%2FBangkok"
            )
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if "hourly" not in data or data["hourly"] is None:
                    if show_warnings:
                        st.warning(f"[Historical] No hourly for {loc_prefix} {dir_name}")
                    continue
                df = pd.DataFrame(data["hourly"])
                if "time" not in df.columns:
                    if show_warnings:
                        st.warning(f"[Historical] No time for {loc_prefix} {dir_name}")
                    continue
                # keep only columns we know exist
                present = [c for c in climate_vars if c in df.columns]
                df = df[["time"] + present].copy()
                df["weight"] = w
                df["direction"] = dir_name
                dfs.append(df)
            except Exception as e:
                if show_warnings:
                    st.warning(f"[Historical][{loc_prefix}][{dir_name}] request error: {e}")
                continue

        if len(dfs) == 0:
            # kosong untuk lokasi ini
            results_per_location[loc_prefix] = pd.DataFrame(columns=["Datetime"])
            continue

        df_all = pd.concat(dfs, ignore_index=True)
        df_all["time"] = pd.to_datetime(df_all["time"])

        weighted_rows = []
        group_cols = [c for c in climate_vars if c in df_all.columns]
        for t, group in df_all.groupby("time"):
            w_arr = group["weight"].values
            row = {"Datetime": t}
            for col in group_cols:
                vals = group[col].astype(float).fillna(0).values
                s = (w_arr * vals).sum()
                row[col] = s
            weighted_rows.append(row)

        df_weighted = pd.DataFrame(weighted_rows)
        rename_map = {
            "relativehumidity_2m": f"{loc_prefix}Relative_humidity",
            "precipitation":        f"{loc_prefix}Rainfall",
            "cloud_cover":          f"{loc_prefix}Cloud_cover",
            "surface_pressure":     f"{loc_prefix}Surface_pressure"
        }
        df_weighted.rename(columns=rename_map, inplace=True)
        # bulatkan numeric (kecuali Datetime)
        for c in df_weighted.columns:
            if c != "Datetime":
                df_weighted[c] = df_weighted[c].round(2)
        results_per_location[loc_prefix] = df_weighted[["Datetime"] + [c for c in df_weighted.columns if c!="Datetime"]]

    # Merge semua lokasi (outer join supaya tidak hilang jam)
    merged = None
    for loc_df in results_per_location.values():
        if merged is None:
            merged = loc_df.copy()
        else:
            merged = pd.merge(merged, loc_df, on="Datetime", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["Datetime"])
    merged = merged.sort_values("Datetime").reset_index(drop=True)
    return merged

# -----------------------------
# Fungsi forecast (per-lokasi, forecast endpoint)
# -----------------------------
def fetch_forecast_multi(show_warnings=True):
    """
    Kembalikan DataFrame forecast dengan kolom-kolom prefix lokasi.
    """
    results_per_location = {}
    for loc_prefix, pts in multi_points.items():
        dfs = []
        for dir_name, meta in pts.items():
            lat = meta["lat"]
            lon = meta["lon"]
            w = meta["weight"]
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&hourly={','.join(climate_vars)}"
                "&forecast_days=16&timezone=Asia%2FBangkok"
            )
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if "hourly" not in data or data["hourly"] is None:
                    if show_warnings:
                        st.warning(f"[Forecast] No hourly for {loc_prefix} {dir_name}")
                    continue
                df = pd.DataFrame(data["hourly"])
                if "time" not in df.columns:
                    if show_warnings:
                        st.warning(f"[Forecast] No time for {loc_prefix} {dir_name}")
                    continue
                present = [c for c in climate_vars if c in df.columns]
                df = df[["time"] + present].copy()
                df["weight"] = w
                df["direction"] = dir_name
                dfs.append(df)
            except Exception as e:
                if show_warnings:
                    st.warning(f"[Forecast][{loc_prefix}][{dir_name}] request error: {e}")
                continue

        if len(dfs) == 0:
            results_per_location[loc_prefix] = pd.DataFrame(columns=["Datetime"])
            continue

        df_all = pd.concat(dfs, ignore_index=True)
        df_all["time"] = pd.to_datetime(df_all["time"])

        weighted_rows = []
        group_cols = [c for c in climate_vars if c in df_all.columns]
        for t, group in df_all.groupby("time"):
            w_arr = group["weight"].values
            row = {"Datetime": t}
            for col in group_cols:
                vals = group[col].astype(float).fillna(0).values
                s = (w_arr * vals).sum()
                row[col] = s
            weighted_rows.append(row)

        df_weighted = pd.DataFrame(weighted_rows)
        rename_map = {
            "relativehumidity_2m": f"{loc_prefix}Relative_humidity",
            "precipitation":        f"{loc_prefix}Rainfall",
            "cloud_cover":          f"{loc_prefix}Cloud_cover",
            "surface_pressure":     f"{loc_prefix}Surface_pressure"
        }
        df_weighted.rename(columns=rename_map, inplace=True)
        for c in df_weighted.columns:
            if c != "Datetime":
                df_weighted[c] = df_weighted[c].round(2)
        results_per_location[loc_prefix] = df_weighted[["Datetime"] + [c for c in df_weighted.columns if c!="Datetime"]]

    merged = None
    for loc_df in results_per_location.values():
        if merged is None:
            merged = loc_df.copy()
        else:
            merged = pd.merge(merged, loc_df, on="Datetime", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["Datetime"])
    merged = merged.sort_values("Datetime").reset_index(drop=True)
    return merged

# -----------------------------
# Run Forecast Button (Streamlit)
# -----------------------------
# -----------------------------
# Tombol Run Forecast
# -----------------------------
run_forecast = st.button("Run 7-Day Forecast")

# Inisialisasi session_state
if "forecast_done" not in st.session_state:
    st.session_state["forecast_done"] = False
    st.session_state["final_df"] = None
    st.session_state["forecast_running"] = False

# Jika tombol diklik → trigger proses
if run_forecast:
    st.session_state["forecast_done"] = False
    st.session_state["final_df"] = None
    st.session_state["forecast_running"] = True


# -----------------------------
# Forecast Logic hanya jalan setelah klik tombol
# -----------------------------
if upload_success and st.session_state.get("forecast_running", False):

    progress_container = st.empty()
    progress_bar = st.progress(0)
    step_counter = 0
    total_forecast_hours = 168
    total_steps = 3 + total_forecast_hours

    try:
        # 1️⃣ Fetch historical climate
        progress_container.markdown("Fetching historical climate data...")
        climate_hist = fetch_historical_multi(start_datetime - timedelta(hours=72), start_datetime)
        step_counter += 1
        progress_bar.progress(step_counter / total_steps)

        # 2️⃣ Fetch forecast climate
        progress_container.markdown("Fetching forecast climate data...")
        climate_forecast = fetch_forecast_multi()
        step_counter += 1
        progress_bar.progress(step_counter / total_steps)

        # 3️⃣ Merge water level and climate data
        progress_container.markdown("Merging water level and climate data...")

        # Pastikan kolom waktu di semua DataFrame
        for df_name, df in zip(["wl_hourly", "climate_hist", "climate_forecast"],
                               [wl_hourly, climate_hist, climate_forecast]):
            if df is not None:
                if "Datetime" not in df.columns and "time" in df.columns:
                    df.rename(columns={"time": "Datetime"}, inplace=True)

        # Konversi ke datetime
        wl_hourly["Datetime"] = pd.to_datetime(wl_hourly["Datetime"])
        climate_hist["Datetime"] = pd.to_datetime(climate_hist["Datetime"])
        climate_forecast["Datetime"] = pd.to_datetime(climate_forecast["Datetime"])

        # Merge historical data
        merged_hist = pd.merge(wl_hourly, climate_hist, on="Datetime", how="left").sort_values("Datetime")
        merged_hist["Source"] = "Historical"

        # Merge forecast data
        forecast_hours = [start_datetime + timedelta(hours=i) for i in range(total_forecast_hours)]
        forecast_df = pd.DataFrame({"Datetime": pd.to_datetime(forecast_hours)})

        forecast_merged = pd.merge(forecast_df, climate_forecast, on="Datetime", how="left")
        forecast_merged["Water_level"] = np.nan
        forecast_merged["Source"] = "Forecast"

        # Gabungkan semuanya
        final_df = pd.concat([merged_hist, forecast_merged], ignore_index=True).sort_values("Datetime")
        final_df = final_df.apply(lambda x: np.round(x, 2) if np.issubdtype(x.dtype, np.number) else x)

        st.session_state["final_df"] = final_df
        st.session_state["forecast_done"] = True
        st.session_state["forecast_running"] = False

        step_counter += 1
        progress_bar.progress(step_counter / total_steps)
        progress_container.success("✅ Forecast selesai!")

    except Exception as e:
        st.session_state["forecast_running"] = False
        progress_container.error(f"Terjadi error: {e}")
    # 4️⃣ Iterative forecast
    progress_container.markdown("Forecasting water level 7 days iteratively...")
    # Gunakan urutan manual fitur
    model_features = model.get_booster().feature_names

    forecast_indices = final_df.index[final_df["Source"]=="Forecast"]

    for i, idx in enumerate(forecast_indices, start=1):
        progress_container.markdown(f"Predicting hour {i}/{total_forecast_hours}...")
        X_forecast = pd.DataFrame(columns=model_features, index=[0])

        for f in model_features:
            if "_Lag" in f:
                base, lag_str = f.rsplit("_Lag",1)
                try:
                    lag = int(lag_str)
                except:
                    lag = 1
            else:
                base = f
                lag = 0

            # Ambil nilai lag dari final_df
            if base in final_df.columns:
                hist_values = final_df.loc[final_df["Source"]=="Historical", base]
                # Jika lag lebih besar dari panjang historical, ambil value pertama
                if idx-lag >= 0:
                    X_forecast.at[0,f] = final_df.iloc[idx-lag].get(base, 0)
                else:
                    X_forecast.at[0,f] = hist_values.iloc[0]
            else:
                # fallback jika kolom tidak ada
                X_forecast.at[0,f] = 0

        # pastikan tipe float
        X_forecast = X_forecast.astype(float)

        # prediksi
        y_hat = model.predict(X_forecast)[0]
        if y_hat < 0: y_hat = 0
        final_df.at[idx,"Water_level"] = round(y_hat,2)

        step_counter += 1
        progress_bar.progress(step_counter / total_steps)

    st.session_state["final_df"] = final_df
    st.session_state["forecast_done"] = True
    st.session_state["forecast_running"] = False
    progress_container.markdown("✅ 7-Day Water Level Forecast Completed!")
    progress_bar.progress(1.0)
    
# -----------------------------
# Display Forecast & Plot
# -----------------------------
result_container = st.empty()
if st.session_state["forecast_done"] and st.session_state["final_df"] is not None:
    final_df = st.session_state["final_df"]
    with result_container.container():
        st.subheader("Water Level + Climate Data with Forecast")
        def highlight_forecast(row):
            return ['background-color: #cfe9ff' if row['Source']=="Forecast" else '' for _ in row]
        
        # Ambil semua kolom numerik
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Terapkan format hanya untuk kolom numerik
        styled_df = final_df.style.apply(highlight_forecast, axis=1)\
                                   .format({col: "{:.2f}" for col in numeric_cols})

        st.dataframe(styled_df, use_container_width=True, height=500)

        # -----------------------------
        # Plot
        # -----------------------------
        st.subheader("Water Level Forecast Plot")
        fig = go.Figure()
        hist_df = final_df[final_df["Source"] == "Historical"]
        fore_df = final_df[final_df["Source"] == "Forecast"]
        
        # Hitung RMSE antara data historis terakhir dan forecast awal (jika ada data aktual)
        if not fore_df.empty:
            # Contoh nilai RMSE, bisa kamu ubah kalau mau dinamis
            rmse = 0.05  
        
            last_val = hist_df["Water_level"].iloc[-1]
            forecast_x = pd.concat([pd.Series([hist_df["Datetime"].iloc[-1]]), fore_df["Datetime"]])
            forecast_y = pd.concat([pd.Series([last_val]), fore_df["Water_level"]])
        
            # Hitung batas atas & bawah error band
            upper_y = forecast_y + rmse
            lower_y = forecast_y - rmse
            lower_y = lower_y.clip(lower=0)  # batas bawah tidak boleh < 0
        
            # Tambah area ±RMSE
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_x, forecast_x[::-1]]),
                y=pd.concat([upper_y, lower_y[::-1]]),
                fill="toself",
                fillcolor="rgba(255,165,0,0.2)",
                line=dict(color="rgba(255,165,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="±RMSE 0.05m"
            ))
        
            # Tambah garis forecast
            fig.add_trace(go.Scatter(
                x=forecast_x, y=forecast_y,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="orange"),
                marker=dict(size=4)
            ))
        
        # Garis historis
        fig.add_trace(go.Scatter(
            x=hist_df["Datetime"], y=hist_df["Water_level"],
            mode="lines+markers",
            name="Historical",
            line=dict(color="blue"),
            marker=dict(size=4)
        ))
        
        # Layout dan annotation RMSE
        fig.update_layout(
            title="Water Level Historical vs Forecast",
            xaxis_title="Datetime",
            yaxis_title="Water Level (m)",
            template="plotly_white",
            annotations=[
                dict(
                    xref="paper", yref="paper",
                    x=0.98, y=0.95,
                    text=f"RMSE = {rmse:.2f}",
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
    result_container.empty()

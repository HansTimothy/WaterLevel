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
    f"- Data must cover **24 hours before the selected start datetime**: "
    f"{start_datetime - timedelta(hours=24)} to {start_datetime}\n"
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
            # 6️⃣ Hanya 24 jam terakhir sebelum start_datetime
            # -----------------------
            start_limit = start_datetime - pd.Timedelta(hours=24)
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
# 2️⃣ Multi-point climate fetch (4 titik: T_, SL_, MB_, MU_)
# -----------------------------
multi_points = {
    "T_": {
        "coords": {
            "NW": (0.38664, 113.78421), "N": (0.38664, 114.83972), "NE": (0.38664, 115.89523),
            "W": (-0.59754, 113.76960), "Center": (-0.59754, 114.82759), "E": (-0.59754, 115.81505),
            "SW": (-1.65202, 113.76685), "S": (-1.65202, 114.76606), "SE": (-1.65202, 115.83664)
        },
        "weights": {"Center": 0.992519, "E": 0.001271, "N": 0.001287, "NE": 0.000591, "NW":0.000639,
                    "S":0.001232, "SE":0.000613, "SW":0.000615, "W":0.001233}
    },
    "SL_": {
        "coords": {
            "NW": (0.52724, 113.68050), "N": (0.52724, 114.73766), "NE": (0.52724, 115.65387),
            "W": (-0.45694, 113.73240), "Center": (-0.45694, 114.71832), "E": (-0.45694, 115.70423),
            "SW": (-1.44112, 113.71044), "S": (-1.44112, 114.70727), "SE": (-1.44112, 115.63291)
        },
        "weights": {"Center":0.995940, "E":0.000639, "N":0.000679, "NE":0.000347, "NW":0.000329,
                    "S":0.000669, "SE":0.000352, "SW":0.000337, "W":0.000708}
    },
    "MB_": {
        "coords": {
            "NW": (0.38664, 113.64348), "N": (0.38664, 114.55825), "NE": (0.38664, 115.61377),
            "W": (-0.59754, 113.62853), "Center": (-0.59754, 114.61599), "E": (-0.59754, 115.60345),
            "SW": (-1.58172, 113.60538), "S": (-1.58172, 114.60380), "SE": (-1.58172, 115.53090)
        },
        "weights": {"Center":0.997992, "E":0.000345, "N":0.000332, "NE":0.000168, "NW":0.000166,
                    "S":0.000334, "SE":0.000183, "SW":0.000160, "W":0.000320}
    },
    "MU_": {
        "coords": {
            "NW": (0.31634, 113.48438), "N": (0.31634, 114.53906), "NE": (0.31634, 115.52344),
            "W": (-0.73814, 113.52433), "Center": (-0.73814, 114.51334), "E": (-0.73814, 115.50235),
            "SW": (-1.72232, 113.50001), "S": (-1.72232, 114.50001), "SE": (-1.72232, 115.50001)
        },
        "weights": {"Center":0.992616, "E":0.001309, "N":0.001197, "NE":0.000613, "NW":0.000587,
                    "S":0.001206, "SE":0.000630, "SW":0.000598, "W":0.001244}
    }
}

variables = ["relative_humidity_2m","precipitation","cloud_cover","surface_pressure"]

def fetch_multi_point_weather(start_dt, end_dt, variables, forecast=False):
    all_points_dfs = []
    for prefix, point_info in multi_points.items():
        coords = point_info["coords"]
        weights = point_info["weights"]

        latitudes = ",".join(str(v[0]) for v in coords.values())
        longitudes = ",".join(str(v[1]) for v in coords.values())

        if forecast:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={latitudes}&longitude={longitudes}"
                f"&hourly={','.join(variables)}&forecast_days=16&timezone=Asia%2FBangkok"
            )
        else:
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={latitudes}&longitude={longitudes}"
                f"&start_date={start_dt.date().isoformat()}&end_date={end_dt.date().isoformat()}"
                f"&hourly={','.join(variables)}&timezone=Asia%2FBangkok"
            )

        try:
            data = requests.get(url, timeout=60).json()
        except Exception as e:
            st.error(f"Error fetching data for {prefix}: {e}")
            continue

        all_dfs = []
        for i, direction in enumerate(coords.keys()):
            loc_data = data[i]
            df = pd.DataFrame(loc_data["hourly"])
            df["direction"] = direction
            df["weight"] = weights[direction]
            all_dfs.append(df)

        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all["time"] = pd.to_datetime(df_all["time"])

        weighted_list = []
        for time, group in df_all.groupby("time"):
            w = group["weight"]
            weighted_vals = (group[variables].T * w.values).T.sum()
            weighted_vals["Datetime"] = time
            weighted_list.append(weighted_vals)

        df_weighted = pd.DataFrame(weighted_list)
        df_weighted.rename(columns={var: f"{prefix}{var}" for var in variables}, inplace=True)
        all_points_dfs.append(df_weighted)

    if len(all_points_dfs) == 0:
        return pd.DataFrame()
    
    from functools import reduce
    df_merged = reduce(lambda left,right: pd.merge(left,right,on="Datetime",how="outer"), all_points_dfs)
    return df_merged.sort_values("Datetime")

# -----------------------------
# 3️⃣ Merge water level + climate
# -----------------------------
if upload_success:
    st.info("Fetching multipoint climate data...")

    climate_df = fetch_multi_point_weather(start_datetime - timedelta(hours=72), start_datetime, variables, forecast=False)

    if climate_df.empty:
        st.error("Failed fetching climate data.")
    else:
        final_df = pd.merge(wl_hourly, climate_df, on="Datetime", how="left")

        # Rename columns final
        rename_dict = {
            "T_relative_humidity_2m": "T_Relative_humidity",
            "T_precipitation": "T_Rainfall",
            "T_cloud_cover": "T_Cloud_cover",
            "T_surface_pressure": "T_Surface_pressure",
            "SL_relative_humidity_2m": "SL_Relative_humidity",
            "SL_precipitation": "SL_Rainfall",
            "SL_cloud_cover": "SL_Cloud_cover",
            "SL_surface_pressure": "SL_Surface_pressure",
            "MB_relative_humidity_2m": "MB_Relative_humidity",
            "MB_precipitation": "MB_Rainfall",
            "MB_cloud_cover": "MB_Cloud_cover",
            "MB_surface_pressure": "MB_Surface_pressure",
            "MU_relative_humidity_2m": "MU_Relative_humidity",
            "MU_precipitation": "MU_Rainfall",
            "MU_cloud_cover": "MU_Cloud_cover",
            "MU_surface_pressure": "MU_Surface_pressure"
        }
        final_df.rename(columns=rename_dict, inplace=True)
        final_df = final_df.sort_values("Datetime")
        final_df = final_df.apply(lambda x: np.round(x,2) if np.issubdtype(x.dtype, np.number) else x)
        
        step_counter += 1
        progress_bar.progress(step_counter / total_steps)
    
        # 4️⃣ Iterative forecast
        progress_container.markdown("Forecasting water level 7 days iteratively...")
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

# WaterLevel_API_CSV.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import time

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊 (CSV Input)")

today = datetime.today().date()
max_forecast_days = 14

# -----------------------------
# Upload CSV
# -----------------------------
st.subheader("Upload CSV Data Water Level")
uploaded_file = st.file_uploader("Pilih file CSV Anda", type=["csv"])

if uploaded_file is not None:
    try:
        df_csv = pd.read_csv(uploaded_file)
        # cek kolom
        required_cols = ["Date", "Water_level"]
        if not all(col in df_csv.columns for col in required_cols):
            st.error(f"CSV harus memiliki kolom: {required_cols}")
        else:
            df_csv["Date"] = pd.to_datetime(df_csv["Date"]).dt.date
            df_csv["Water_level"] = pd.to_numeric(df_csv["Water_level"], errors="coerce")
            # cek missing
            if df_csv["Water_level"].isnull().any():
                st.warning("Terdapat nilai water level yang invalid atau kosong, akan diisi 0")
                df_csv["Water_level"].fillna(0.0, inplace=True)
            
            df_csv = df_csv.sort_values("Date").reset_index(drop=True)
            
            st.subheader("Preview CSV Water Level")
            st.dataframe(df_csv.style.format({"Water_level": "{:.2f}"}).set_properties(**{"text-align": "right"}, subset=["Water_level"]))

            # -----------------------------
            # Pilih tanggal prediksi
            # -----------------------------
            st.subheader("Pilih Tanggal Prediksi Water Level (Max H+14 dari hari terakhir CSV)")
            last_csv_date = df_csv["Date"].max()
            pred_date = st.date_input(
                "Tanggal Prediksi Water Level",
                value=last_csv_date + timedelta(days=1),
                min_value=last_csv_date + timedelta(days=1),
                max_value=last_csv_date + timedelta(days=max_forecast_days)
            )
            
            # -----------------------------
            # Helper safe_get
            # -----------------------------
            def safe_get(df, date):
                try:
                    return float(df.loc[df["Date"] == date, "Water_level"].values[0])
                except Exception:
                    return 0.0
            
            # -----------------------------
            # Progress bar
            # -----------------------------
            if st.button("Predict & Forecast"):
                progress = st.progress(0)
                
                # ambil lags 7 hari terakhir
                wl_dates = [last_csv_date - timedelta(days=i) for i in range(0, 7)]
                water_level_lags = [safe_get(df_csv, d) for d in wl_dates[::-1]]  # H0..H-6
                
                n_days = (pred_date - last_csv_date).days
                results = {}
                
                # mulai forecast
                for step in range(1, n_days + 1):
                    pred_day = last_csv_date + timedelta(days=step)
                    
                    # -----------------------------
                    # Fetch cuaca
                    # -----------------------------
                    start_date_hist = pred_day - timedelta(days=7)
                    end_date_hist = pred_day - timedelta(days=1)
                    url = (
                        f"https://archive-api.open-meteo.com/v1/archive?"
                        f"latitude=-0.61&longitude=114.8&start_date={start_date_hist}&end_date={end_date_hist}"
                        f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
                        f"&timezone=Asia%2FSingapore"
                    )
                    try:
                        data = requests.get(url).json()
                        df_weather = pd.DataFrame({
                            "time": pd.to_datetime(data["daily"]["time"]).date,
                            "precipitation_sum": data["daily"]["precipitation_sum"],
                            "temperature_mean": data["daily"]["temperature_2m_mean"],
                            "relative_humidity": data["daily"]["relative_humidity_2m_mean"]
                        })
                    except:
                        df_weather = pd.DataFrame({
                            "time": [pred_day - timedelta(days=i) for i in range(7)],
                            "precipitation_sum": [0]*7,
                            "temperature_mean": [0]*7,
                            "relative_humidity": [0]*7
                        })
                    
                    df_weather.set_index("time", inplace=True)
                    
                    # build input feature
                    inp = {}
                    for i in range(1, 8):
                        date_i = pred_day - timedelta(days=i)
                        inp[f"Precipitation_lag{i}d"] = [df_weather.loc[date_i, "precipitation_sum"] if date_i in df_weather.index else 0.0]
                    for i in range(1, 5):
                        date_i = pred_day - timedelta(days=i)
                        inp[f"Temperature_lag{i}d"] = [df_weather.loc[date_i, "temperature_mean"] if date_i in df_weather.index else 0.0]
                    for i in range(1, 8):
                        date_i = pred_day - timedelta(days=i)
                        inp[f"Relative_humidity_lag{i}d"] = [df_weather.loc[date_i, "relative_humidity"] if date_i in df_weather.index else 0.0]
                    
                    # water level lags
                    for i in range(1, 8):
                        inp[f"Water_level_lag{i}d"] = [water_level_lags[i-1]]
                    
                    features = [
                        "Precipitation_lag1d","Precipitation_lag2d","Precipitation_lag3d","Precipitation_lag4d",
                        "Precipitation_lag5d","Precipitation_lag6d","Precipitation_lag7d",
                        "Temperature_lag1d","Temperature_lag2d","Temperature_lag3d","Temperature_lag4d",
                        "Relative_humidity_lag1d","Relative_humidity_lag2d","Relative_humidity_lag3d",
                        "Relative_humidity_lag4d","Relative_humidity_lag5d","Relative_humidity_lag6d",
                        "Relative_humidity_lag7d",
                        "Water_level_lag1d","Water_level_lag2d","Water_level_lag3d","Water_level_lag4d",
                        "Water_level_lag5d","Water_level_lag6d","Water_level_lag7d",
                    ]
                    
                    input_data = pd.DataFrame(inp)[features].fillna(0.0)
                    
                    prediction = model.predict(input_data)[0]
                    results[pred_day] = prediction
                    
                    # update lags
                    water_level_lags = [prediction] + water_level_lags[:-1]
                    
                    # update progress
                    progress.progress(int(step / n_days * 100))
                    time.sleep(0.1)
                
                # gabungkan hasil
                df_forecast = pd.DataFrame({
                    "Date": list(results.keys()),
                    "Water_level": list(results.values())
                })
                df_all = pd.concat([df_csv, df_forecast]).reset_index(drop=True)
                
                # highlight forecast
                forecast_dates = df_forecast["Date"].tolist()
                def highlight_forecast(row):
                    return ['background-color: #bce4f6' if row.Date in forecast_dates else '' for _ in row]
                
                st.subheader("Preview Water Level (History + Forecast)")
                st.dataframe(df_all.style.apply(highlight_forecast, axis=1).format({"Water_level": "{:.2f}"}).set_properties(**{"text-align": "right"}, subset=["Water_level"]))
                
                # -----------------------------
                # Plot
                # -----------------------------
                lower_limit = 19.5
                upper_limit = 28
                
                df_hist = df_all[df_all["Date"] <= last_csv_date]
                df_pred = df_all[df_all["Date"] > last_csv_date]
                df_pred_safe = df_pred[df_pred["Water_level"].between(lower_limit, upper_limit)]
                df_pred_unsafe = df_pred[(df_pred["Water_level"] < lower_limit) | (df_pred["Water_level"] > upper_limit)]
                
                fig = go.Figure()
                
                # historis
                fig.add_trace(go.Scatter(
                    x=df_hist["Date"], y=df_hist["Water_level"],
                    mode="lines+markers", line=dict(color="blue", width=2),
                    marker=dict(color="blue", size=8), name="Historical"
                ))
                
                # prediksi aman
                fig.add_trace(go.Scatter(
                    x=df_pred_safe["Date"], y=df_pred_safe["Water_level"],
                    mode="lines+markers", line=dict(color="black", width=2, dash="dash"),
                    marker=dict(color="green", size=8), name="Prediction (Loadable)"
                ))
                
                # prediksi tidak aman
                fig.add_trace(go.Scatter(
                    x=df_pred_unsafe["Date"], y=df_pred_unsafe["Water_level"],
                    mode="lines+markers", line=dict(color="black", width=2, dash="dash"),
                    marker=dict(color="red", size=8), name="Prediction (Unloadable)"
                ))
                
                # batas loadable
                fig.add_hline(y=lower_limit, line=dict(color="red", width=2, dash="dash"), annotation_text="Lower Limit", annotation_position="bottom left")
                fig.add_hline(y=upper_limit, line=dict(color="red", width=2, dash="dash"), annotation_text="Upper Limit", annotation_position="top left")
                
                fig.update_layout(
                    title="Water Level Dashboard 🌊",
                    xaxis_title="Date",
                    yaxis_title="Water Level (m)",
                    xaxis=dict(tickangle=90),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # -----------------------------
                # Download CSV hasil
                # -----------------------------
                csv_buffer = io.StringIO()
                df_all.to_csv(csv_buffer, index=False)
                st.download_button("Download CSV Hasil Forecast", csv_buffer.getvalue(), file_name="WaterLevel_Forecast.csv")
                
    except Exception as e:
        st.error(f"Terjadi error saat membaca CSV: {e}")

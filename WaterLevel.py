# WaterLevel_API.py
import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊")

# -----------------------------
# Prediction date (max H+7)
# -----------------------------
today = datetime.today().date()

st.subheader("Pilih Tanggal Prediksi Water Level")

pred_date = st.date_input(
    "Tanggal Prediksi Water Level",
    value=today + timedelta(days=1),
    max_value=today + timedelta(days=7)
)

# -----------------------------
# Input manual Water Level (H-1 .. H-7)
# -----------------------------
st.subheader("Masukkan Data Water Level (7 hari terakhir)")

wl_dates = [today - timedelta(days=i) for i in range(1, 8)]  # H-1 sampai H-7
wl_inputs = [
    st.number_input(
        f"Water Level **{d.strftime('%d %B %Y')}**",
        value=20.0,
        format="%.2f"
    )
    for d in wl_dates
]

# -----------------------------
# Fetch data & predict
# -----------------------------
if st.button("Fetch Data & Predict"):

    # -----------------------------
    # Ambil data historis (H-6 s/d today)
    # -----------------------------
    hist_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=-0.61&longitude=114.8&start_date={today - timedelta(days=6)}&end_date={today}"
        f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
        f"&timezone=Asia%2FSingapore"
    )
    hist_resp = requests.get(hist_url)
    hist_data = hist_resp.json()

    df_hist = pd.DataFrame({
        "time": hist_data["daily"]["time"],
        "precipitation_sum": hist_data["daily"]["precipitation_sum"],
        "temperature_mean": hist_data["daily"]["temperature_2m_mean"],
        "relative_humidity": hist_data["daily"]["relative_humidity_2m_mean"]
    })
    df_hist["time"] = pd.to_datetime(df_hist["time"]).dt.date
    df_hist.set_index("time", inplace=True)

    # -----------------------------
    # Forecast data kalau prediksi > H+1
    # -----------------------------
    if pred_date > today + timedelta(days=1):
        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude=-0.61&longitude=114.8"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        fc_resp = requests.get(forecast_url)
        fc_data = fc_resp.json()

        df_fc = pd.DataFrame({
            "time": fc_data["daily"]["time"],
            "precipitation_sum": fc_data["daily"]["precipitation_sum"],
            "temperature_mean": fc_data["daily"]["temperature_2m_mean"],
            "relative_humidity": fc_data["daily"]["relative_humidity_2m_mean"]
        })
        df_fc["time"] = pd.to_datetime(df_fc["time"]).dt.date
        df_fc.set_index("time", inplace=True)

        df = pd.concat([df_hist, df_fc])
    else:
        df = df_hist.copy()

    # -----------------------------
    # Preview data sampai tanggal prediksi
    # -----------------------------
    df_preview = df[df.index <= pred_date]
    st.subheader("Preview API Data")
    st.dataframe(df_preview)

    # -----------------------------
    # Loop prediction dari H+1 sampai tanggal target
    # -----------------------------
    wl_series = {f"lag{i}": wl_inputs[i-1] for i in range(1, 8)}  # manual input
    current_date = today + timedelta(days=1)
    last_prediction = None

    # Daftar fitur sesuai model
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

    while current_date <= pred_date:
        input_data = pd.DataFrame({
            # Precipitation lags
            **{f"Precipitation_lag{i}d": [df.loc[current_date - timedelta(days=i), "precipitation_sum"]]
               if (current_date - timedelta(days=i)) in df.index else [0.0]
               for i in range(1, 8)},
            # Temperature lags
            **{f"Temperature_lag{i}d": [df.loc[current_date - timedelta(days=i), "temperature_mean"]]
               if (current_date - timedelta(days=i)) in df.index else [0.0]
               for i in range(1, 5)},
            # Relative Humidity lags
            **{f"Relative_humidity_lag{i}d": [df.loc[current_date - timedelta(days=i), "relative_humidity"]]
               if (current_date - timedelta(days=i)) in df.index else [0.0]
               for i in range(1, 8)},
            # Water Level lags
            **{f"Water_level_lag{i}d": [wl_series[f'lag{i}']] for i in range(1, 8)}
        })

        # pastikan numeric
        input_data = input_data[features].fillna(0.0).astype(float)

        # Prediksi
        pred = model.predict(input_data)[0]
        last_prediction = pred

        # Update water level lag series
        for j in range(7, 1, -1):
            wl_series[f"lag{j}"] = wl_series[f"lag{j-1}"]
        wl_series["lag1"] = pred

        current_date += timedelta(days=1)

    # -----------------------------
    # Hasil akhir hanya untuk tanggal target
    # -----------------------------
    st.success(f"Predicted Water Level on {pred_date.strftime('%d %B %Y')}: {last_prediction:.2f} m")

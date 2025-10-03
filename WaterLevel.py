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

today = datetime.today().date()

st.subheader("Pilih Tanggal Prediksi Water Level")

pred_date = st.date_input(
    "Tanggal Prediksi Water Level",
    value=today + timedelta(days=1),   
    min_value=today - timedelta(days=30),  # bisa pilih masa lalu
    max_value=today + timedelta(days=7)    # atau sampai H+7
)

# -----------------------------
# Input manual Water Level
# -----------------------------
st.subheader("Masukkan Data Water Level")

if pred_date <= today + timedelta(days=1):
    # prediksi masa lalu atau H+1
    wl_dates = [pred_date - timedelta(days=i) for i in range(0, 7)]
else:
    # prediksi H+2..H+7 → pakai data hari ini sampai H-6
    wl_dates = [today - timedelta(days=i) for i in range(0, 7)]

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

    if pred_date <= today + timedelta(days=1):
        # --- Skenario 1: prediksi masa lalu atau H+1 ---
        start_date = pred_date - timedelta(days=7)
        end_date = pred_date - timedelta(days=1)

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=-0.61&longitude=114.8&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        data = requests.get(url).json()

        df = pd.DataFrame({
            "time": data["daily"]["time"],
            "precipitation_sum": data["daily"]["precipitation_sum"],
            "temperature_mean": data["daily"]["temperature_2m_mean"],
            "relative_humidity": data["daily"]["relative_humidity_2m_mean"]
        })
        df["time"] = pd.to_datetime(df["time"]).dt.date
        df.set_index("time", inplace=True)

        st.subheader("Preview Data Historis")
        st.dataframe(df)

        # buat input feature
        input_data = pd.DataFrame({
            **{f"Precipitation_lag{i}d": [df["precipitation_sum"].iloc[-i]] for i in range(1, 8)},
            **{f"Temperature_lag{i}d": [df["temperature_mean"].iloc[-i]] for i in range(1, 5)},
            **{f"Relative_humidity_lag{i}d": [df["relative_humidity"].iloc[-i]] for i in range(1, 8)},
            **{f"Water_level_lag{i}d": [wl_inputs[i]] for i in range(1, 8)}
        })

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
        input_data = input_data[features].fillna(0.0)

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Water Level on {pred_date.strftime('%d %B %Y')}: {prediction:.2f} m")

    else:
        # --- Skenario 2: prediksi H+2..H+7 ---
        # Data historis H-6 .. H0
        url_hist = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=-0.61&longitude=114.8&start_date={today - timedelta(days=6)}&end_date={today}"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        hist = requests.get(url_hist).json()
        df_hist = pd.DataFrame({
            "time": hist["daily"]["time"],
            "precipitation_sum": hist["daily"]["precipitation_sum"],
            "temperature_mean": hist["daily"]["temperature_2m_mean"],
            "relative_humidity": hist["daily"]["relative_humidity_2m_mean"]
        })
        df_hist["time"] = pd.to_datetime(df_hist["time"]).dt.date

        # Data forecast H+1..H+7
        url_forecast = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude=-0.61&longitude=114.8"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        forecast = requests.get(url_forecast).json()
        df_forecast = pd.DataFrame({
            "time": forecast["daily"]["time"],
            "precipitation_sum": forecast["daily"]["precipitation_sum"],
            "temperature_mean": forecast["daily"]["temperature_2m_mean"],
            "relative_humidity": forecast["daily"]["relative_humidity_2m_mean"]
        })
        df_forecast["time"] = pd.to_datetime(df_forecast["time"]).dt.date

        df = pd.concat([df_hist, df_forecast])
        df.set_index("time", inplace=True)

        st.subheader("Preview Data (Hist + Forecast)")
        st.dataframe(df)

        results = {}
        water_level_lags = wl_inputs[:]  # manual input

        n_days = (pred_date - today).days
        for step in range(1, n_days + 1):
            pred_day = today + timedelta(days=step)

            input_data = pd.DataFrame({
                **{f"Precipitation_lag{i}d": [df.loc[pred_day - timedelta(days=i), "precipitation_sum"]] for i in range(1, 8)},
                **{f"Temperature_lag{i}d": [df.loc[pred_day - timedelta(days=i), "temperature_mean"]] for i in range(1, 5)},
                **{f"Relative_humidity_lag{i}d": [df.loc[pred_day - timedelta(days=i), "relative_humidity"]] for i in range(1, 8)},
                **{f"Water_level_lag{i}d": [water_level_lags[i]] for i in range(1, 8)}
            })

            input_data = input_data[features].fillna(0.0)

            prediction = model.predict(input_data)[0]
            results[pred_day] = prediction

            # update lag
            water_level_lags = [prediction] + water_level_lags[:-1]

        st.subheader("Hasil Prediksi")
        for d, val in results.items():
            st.success(f"Predicted Water Level on {d.strftime('%d %B %Y')}: {val:.2f} m")

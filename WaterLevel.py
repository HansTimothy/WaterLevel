# WaterLevel_API.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊")

# -----------------------------
# Prediction date = H+1
# -----------------------------
today = datetime.today().date()

st.subheader(f"Pilih Tanggal Prediksi Water Level")

pred_date = st.date_input(
    "Tanggal Prediksi Water Level",
    value=today + timedelta(days=1),   # default H+1
    max_value=today + timedelta(days=1)
)

start_date = pred_date - timedelta(days=7)  # H-6
end_date = pred_date - timedelta(days=1)

st.write(f"Prediksi Water Level untuk tanggal **{pred_date.strftime("%d/%m/%Y")}** menggunakan data harian dari Open-Meteo API")
st.write(f"Data API akan diambil dari **{start_date.strftime("%d/%m/%Y")}** sampai **{end_date.strftime("%d %B %Y")}**")

# -----------------------------
# Input manual Water Level H sampai H-6
# -----------------------------
st.subheader(f"Masukkan Data Water Level")

# buat list tanggal H sampai H-6
wl_dates = [pred_date - timedelta(days=i) for i in range(1, 8)]

# input manual dengan tanggal sebagai label (bold)
wl_inputs = [
    st.number_input(
        f" Water Level **{d.strftime("%d/%m/%Y")}**",
        value=20.0,
        format="%.2f"
    )
    for i, d in enumerate(wl_dates)
]

# -----------------------------
# Fetch data & predict
# -----------------------------
if st.button("Fetch Data & Predict"):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=-0.61&longitude=114.8&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
        f"&timezone=Asia%2FSingapore"
    )
    
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame({
        "time": data["daily"]["time"],
        "precipitation_sum": data["daily"]["precipitation_sum"],
        "temperature_mean": data["daily"]["temperature_2m_mean"],
        "relative_humidity": data["daily"]["relative_humidity_2m_mean"]
    })
    
    df["time"] = pd.to_datetime(df["time"]).dt.date
    df.set_index("time", inplace=True)
    
    st.subheader("Preview API Data")
    st.dataframe(df)

    # -----------------------------
    # Buat input dataframe sesuai feature model
    # -----------------------------
    input_data = pd.DataFrame({
        # Precipitation lags
        **{f"Precipitation_lag{i}d": [df["precipitation_sum"].iloc[-i]] for i in range(1, 8)},
        # Temperature lags
        **{f"Temperature_lag{i}d": [df["temperature_mean"].iloc[-i]] for i in range(1, 5)},
        # Relative Humidity lags
        **{f"Relative_humidity_lag{i}d": [df["relative_humidity"].iloc[-i]] for i in range(1, 8)},
        # Water Level lags (manual input)
        **{f"Water_level_lag{i}d": [wl_inputs[i-1]] for i in range(1, 8)}
    })
    
    # pastikan urutan kolom sesuai model
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

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Water Level on {pred_date}: {prediction:.2f} m")

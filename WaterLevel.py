# WaterLevel_API.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import timedelta

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊")
st.write("Prediksi Water Level menggunakan data harian dari Open-Meteo API")

# -----------------------------
# Input manual Water Level Lag 1–7 hari
# -----------------------------
st.subheader("Masukkan Water Level Lag 1–7 hari (manual)")
wl_inputs = [st.number_input(f"Water Level Lag {i}d", value=21.0-i*0.1, step=0.1) for i in range(1, 8)]

# -----------------------------
# Input tanggal prediksi
# -----------------------------
pred_date = st.date_input("Prediction date", pd.to_datetime("2025-09-30"))
start_date = pred_date - timedelta(days=7)
end_date = pred_date

st.write(f"Data API akan diambil dari {start_date} sampai {end_date}")

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
    
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    
    st.subheader("Preview API Data")
    st.dataframe(df)

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
    
    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Water Level on {pred_date}: {prediction:.2f} m")

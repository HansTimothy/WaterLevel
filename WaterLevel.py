# WaterLevel_API.py
import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊")
st.write("Prediksi Water Level menggunakan data harian dari Open-Meteo API")

# -----------------------------
# Pilihan tanggal prediksi
# -----------------------------
pred_date = st.date_input("Tanggal prediksi", value=pd.to_datetime("2025-09-30"))

# -----------------------------
# Ambil data API (7 hari terakhir)
# -----------------------------
latitude = -0.61
longitude = 114.8
end_date = pred_date
start_date = pred_date - timedelta(days=6)  # ambil 7 hari terakhir

api_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean&timezone=Asia%2FSingapore"

st.write("Fetching data from API...")
response = requests.get(api_url)
data = response.json()

# -----------------------------
# Convert API response ke DataFrame
# -----------------------------
df = pd.DataFrame({
    "date": data["daily"]["time"],
    "Precipitation": data["daily"]["precipitation_sum"],
    "Temperature": data["daily"]["temperature_2m_mean"],
    "Relative_humidity": data["daily"]["relative_humidity_2m_mean"]
})
df.set_index("date", inplace=True)

# Dummy Water Level (kalau mau input manual)
st.subheader("Masukkan Water Level Lag 1–7 hari")
wl_inputs = [st.number_input(f"Water Level Lag {i}d", value=21.0-i*0.1, step=0.1) for i in range(1, 8)]

# -----------------------------
# Buat lag harian (1–7 hari)
# -----------------------------
input_data = pd.DataFrame({
    # Precipitation lags
    **{f"Precipitation_lag{i}d": [df["Precipitation"].iloc[-i]] for i in range(1, 8)},
    # Temperature lags
    **{f"Temperature_lag{i}d": [df["Temperature"].iloc[-i]] for i in range(1, 5)},
    # Relative Humidity lags
    **{f"Relative_humidity_lag{i}d": [df["Relative_humidity"].iloc[-i]] for i in range(1, 8)},
    # Water Level lags (manual input)
    **{f"Water_level_lag{i}d": [wl_inputs[i-1]] for i in range(1, 8)}
})

st.write("Input data for prediction:")
st.dataframe(input_data)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Water Level"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Water Level: {prediction:.2f} m")
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊")
st.write("Masukkan nilai variabel harian untuk memprediksi Water Level:")

# -----------------------------
# Input manual untuk setiap variable dan lag
# -----------------------------
st.subheader("Precipitation (mm)")
prec_inputs = [st.number_input(f"Precipitation Lag {i}d", value=200.0-i*5, step=1.0) for i in range(1, 8)]

st.subheader("Temperature (°C)")
temp_inputs = [st.number_input(f"Temperature Lag {i}d", value=0.0, step=0.01) for i in range(1, 5)]

st.subheader("Relative Humidity (%)")
rh_inputs = [st.number_input(f"Relative Humidity Lag {i}d", value=90.0-i, step=0.1) for i in range(1, 8)]

st.subheader("Water Level (m)")
wl_inputs = [st.number_input(f"Water Level Lag {i}d", value=21.0-i*0.1, step=0.1) for i in range(1, 8)]

# -----------------------------
# Prepare input dataframe
# -----------------------------
input_data = pd.DataFrame({
    # Precipitation lags
    **{f"Precipitation_lag{i}d": [prec_inputs[i-1]] for i in range(1, 8)},
    # Temperature lags
    **{f"Temperature_lag{i}d": [temp_inputs[i-1]] for i in range(1, 5)},
    # Relative Humidity lags
    **{f"Relative_humidity_lag{i}d": [rh_inputs[i-1]] for i in range(1, 8)},
    # Water Level lags
    **{f"Water_level_lag{i}d": [wl_inputs[i-1]] for i in range(1, 8)}
})

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Water Level"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Water Level: {prediction:.2f} m")

# WaterLevel_API_fixed.py
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
    min_value=today - timedelta(days=30),
    max_value=today + timedelta(days=7)
)

# features (dipakai di kedua skenario)
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

# -----------------------------
# Tentukan tanggal untuk input Water Level manual
# -----------------------------
st.subheader("Masukkan Data Water Level (manual)")

if pred_date <= today + timedelta(days=1):
    # Prediksi untuk masa lalu atau H+1:
    # kita membutuhkan water level untuk pred_date-1 .. pred_date-7
    wl_dates = [pred_date - timedelta(days=i) for i in range(1, 8)]
else:
    # Prediksi H+2..H+7:
    # user mau memasukkan water level dari hari ini (H0) sampai H-6
    # sehingga wl_inputs[0] = hari ini (paling baru)
    wl_dates = [today - timedelta(days=i) for i in range(0, 7)]

wl_inputs = [
    st.number_input(f"Water Level {d.strftime('%d %B %Y')}", value=20.0, format="%.2f")
    for d in wl_dates
]

# -----------------------------
# Helper untuk ambil value dari df dengan fallback
# -----------------------------
def safe_get(df, date, col):
    try:
        return float(df.loc[date, col])
    except Exception:
        return 0.0

# -----------------------------
# Fetch data & predict
# -----------------------------
if st.button("Fetch Data & Predict"):
    # tentukan apakah pakai data historis atau forecast
    if pred_date <= today + timedelta(days=1):
        # Historical data
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=-0.61&longitude=114.8&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
    else:
        # Forecast data
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude=-0.61&longitude=114.8"
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

    # Jika prediksi > H+1, preview hanya sampai tanggal prediksi
    df_preview = df[df.index <= pred_date]
    st.subheader("Preview API Data")
    st.dataframe(df_preview)

    # -----------------------------
    # Loop prediction
    # -----------------------------
    wl_series = {f"lag{i}": wl_inputs[i-1] for i in range(1, 8)}  # manual input H-1..H-7
    current_date = today + timedelta(days=1)  # mulai prediksi H+1
    last_prediction = None

    while current_date <= pred_date:
        # Buat input sesuai feature model
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
            **{f"Water_level_lag{i}d": [wl_series[f"lag{i}"]] for i in range(1, 8)}
        })

        input_data = input_data[features].fillna(0.0)

        # Prediksi
        pred = model.predict(input_data)[0]
        last_prediction = pred

        # update water level lag series (geser)
        for j in range(7, 1, -1):
            wl_series[f"lag{j}"] = wl_series[f"lag{j-1}"]
        wl_series["lag1"] = pred

        current_date += timedelta(days=1)

    # -----------------------------
    # Tampilkan hanya prediksi akhir sesuai tanggal dipilih
    # -----------------------------
    st.success(f"Predicted Water Level on {pred_date}: {last_prediction:.2f} m")

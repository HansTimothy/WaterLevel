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

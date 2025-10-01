import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊")
st.write("Masukkan nilai variabel berikut untuk memprediksi Water Level:")

# -----------------------------
# Input manual
# -----------------------------

iod = st.number_input("IOD", value=0.1, step=0.01)
prec_lag1 = st.number_input("Precipitation Lag1", value=200.0, step=1.0)
prec_lag2 = st.number_input("Precipitation Lag2", value=180.0, step=1.0)
temp_lag1 = st.number_input("Temp Anomaly Lag1", value=0.05, step=0.01)
temp_lag2 = st.number_input("Temp Anomaly Lag2", value=0.0, step=0.01)
temp_lag3 = st.number_input("Temp Anomaly Lag3", value=-0.02, step=0.01)
wl_lag1 = st.number_input("Water Level Lag1", value=21.0, step=0.1)
wl_lag2 = st.number_input("Water Level Lag2", value=20.8, step=0.1)

# -----------------------------
# Prepare input dataframe
# -----------------------------

input_data = pd.DataFrame({
    "IOD_Lag1": [iod],
    "Precipitation_Lag1": [prec_lag1],
    "Precipitation_Lag2": [prec_lag2],
    "Temp_anomaly_Lag1": [temp_lag1],
    "Temp_anomaly_Lag2": [temp_lag2],
    "Temp_anomaly_Lag3": [temp_lag3],
    "Water_level_Lag1": [wl_lag1],
    "Water_level_Lag2": [wl_lag2]
})

# -----------------------------
# Prediction button
# -----------------------------

if st.button("Predict Water Level"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Water Level: {prediction:.2f}")

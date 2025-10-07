# WaterLevel_API_fixed.py
import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊")

today = datetime.today().date()

# -----------------------------
# Pilihan tanggal prediksi
# -----------------------------
pred_date = st.date_input(
    "Tanggal Prediksi Water Level",
    value=today + timedelta(days=1),
    min_value=datetime(2018, 5, 1).date(),
    max_value=today + timedelta(days=14),
    format="DD/MM/YYYY"
)

# -----------------------------
# Fetch data
# -----------------------------
url = "https://api.example.com/waterlevel"  # ganti dengan API Anda
response = requests.get(url)
df = pd.DataFrame(response.json())

# Pastikan kolom tanggal dalam format date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
df = df.sort_values("Date")

# -----------------------------
# Prediksi
# -----------------------------
# (Simulasi: ubah sesuai model Anda)
df_pred = df.copy()
df_pred["predicted"] = model.predict(df_pred[["feature1", "feature2"]])  # contoh kolom

# Tentukan batas data historis dan prediksi
last_hist_date = df["Date"].max()

# -----------------------------
# Plot
# -----------------------------
fig = go.Figure()

# 1️⃣ Garis historis
fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["water_level"],
    mode="lines",
    name="Historis",
    line=dict(color="blue", width=2)
))

# 2️⃣ Garis prediksi
fig.add_trace(go.Scatter(
    x=df_pred["Date"],
    y=df_pred["predicted"],
    mode="lines",
    name="Prediksi",
    line=dict(color="orange", width=2, dash="dash")
))

# 3️⃣ Garis dashed penghubung antara historis & prediksi
if pred_date > last_hist_date:
    last_hist_value = df.loc[df["Date"] == last_hist_date, "water_level"].values[0]
    first_pred_value = df_pred.loc[df_pred["Date"] == pred_date, "predicted"].values[0]

    fig.add_trace(go.Scatter(
        x=[last_hist_date, pred_date],
        y=[last_hist_value, first_pred_value],
        mode="lines",
        line=dict(color="orange", width=2, dash="dash"),
        showlegend=False
    ))

# 4️⃣ Area RMSE (contoh ±RMSE di sekitar prediksi)
rmse = 0.25  # contoh nilai RMSE, ganti sesuai hasil evaluasi model
fig.add_trace(go.Scatter(
    x=pd.concat([df_pred["Date"], df_pred["Date"][::-1]]),
    y=pd.concat([
        df_pred["predicted"] + rmse,
        (df_pred["predicted"] - rmse)[::-1]
    ]),
    fill="toself",
    fillcolor="rgba(255, 165, 0, 0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Area RMSE",
    showlegend=True
))

# -----------------------------
# Layout
# -----------------------------
fig.update_layout(
    title=f"Prediksi Water Level hingga {pred_date.strftime('%d/%m/%Y')}",
    xaxis_title="Tanggal",
    yaxis_title="Water Level (m)",
    legend=dict(x=0, y=1.1, orientation="h"),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

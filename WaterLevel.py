import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# ===============================
# 🎯 Judul Aplikasi
# ===============================
st.title("🌊 Water Level Forecast Jetty Tuhup")

# ===============================
# 📂 Upload CSV Data Historis
# ===============================
uploaded_file = st.file_uploader("📥 Upload file AWLR Log Jetty Tuhup", type=["csv"])

if uploaded_file is None:
    st.warning("Silakan upload file CSV historis terlebih dahulu.")
    st.stop()

# Baca file CSV
df_hist = pd.read_csv(uploaded_file)
df_hist["Datetime"] = pd.to_datetime(df_hist["Datetime"])
df_hist = df_hist.sort_values("Datetime").reset_index(drop=True)
df_hist.rename(columns={"Level Air": "Water_level"}, inplace=True)

# ===============================
# ⚙️ Load Model
# ===============================
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# ===============================
# 📅 Input Parameter
# ===============================
today = pd.Timestamp(datetime.now().date())
start_date = df_hist["Datetime"].min()
end_date = df_hist["Datetime"].max()

selected_date = st.date_input(
    "📅 Pilih tanggal mulai prediksi (maksimum hari ini):",
    value=min(today, end_date),
    min_value=start_date.date(),
    max_value=today.date()
)
selected_date = pd.Timestamp(selected_date)

st.info(f"🔍 Prediksi akan dimulai dari {selected_date.date()} dan berjalan 14 hari ke depan.")

# ===============================
# 🔗 Ambil Data Iklim Otomatis (Open-Meteo)
# ===============================
st.info("🔄 Mengambil data iklim otomatis dari Open-Meteo API...")

latitude, longitude = -0.1177, 114.1002  # Contoh koordinat Sungai Joloi

start_fetch = (selected_date - timedelta(days=7)).strftime("%Y-%m-%d")
end_fetch = (selected_date + timedelta(days=14)).strftime("%Y-%m-%d")

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_fetch}&end_date={end_fetch}"
    f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
    f"&timezone=Asia/Singapore"
)

r = requests.get(url)
if r.status_code != 200:
    st.error("Gagal mengambil data iklim dari API.")
    st.stop()

data = r.json()
climate_df = pd.DataFrame({
    "Datetime": data["daily"]["time"],
    "Rainfall": data["daily"]["precipitation_sum"],
    "Temp_max": data["daily"]["temperature_2m_max"],
    "Temp_min": data["daily"]["temperature_2m_min"]
})
climate_df["Datetime"] = pd.to_datetime(climate_df["Datetime"])

# ===============================
# 🧮 Siapkan Data untuk Model
# ===============================
full_df = pd.merge(climate_df, df_hist, on="Datetime", how="left")
full_df["Water_level"] = full_df["Water_level"].interpolate()

# Filter hanya data >= selected_date - 7 hari
full_df = full_df[full_df["Datetime"] >= (selected_date - timedelta(days=7))].reset_index(drop=True)

# ===============================
# 🔁 Autoregresi Forecasting (14 hari)
# ===============================
lags = 7
forecast_horizon = 14

# Buat lag awal
for i in range(1, lags + 1):
    full_df[f"WL_Lag{i}"] = full_df["Water_level"].shift(i)

full_df = full_df.sort_values("Datetime").reset_index(drop=True)

# Ambil data terakhir sebelum prediksi
last_known = full_df.dropna().iloc[-1]
history_values = [last_known[f"WL_Lag{i}"] for i in range(1, lags + 1)][::-1]

forecast_rows = []
dates_future = [selected_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]

for d in dates_future:
    if d not in climate_df["Datetime"].values:
        continue

    climate_today = climate_df[climate_df["Datetime"] == d][["Rainfall", "Temp_max", "Temp_min"]].values[0]
    features = np.concatenate((np.array(history_values[-lags:]), climate_today)).reshape(1, -1)
    pred = model.predict(features)[0]
    history_values.append(pred)

    forecast_rows.append({
        "Datetime": d,
        "Predicted_Water_Level": pred
    })

forecast_df = pd.DataFrame(forecast_rows)

# ===============================
# 📊 Gabungkan Hasil
# ===============================
combined_df = pd.concat([
    full_df[["Datetime", "Water_level"]],
    forecast_df.rename(columns={"Predicted_Water_Level": "Water_level"})
])
combined_df["Source"] = np.where(combined_df["Datetime"] < forecast_df["Datetime"].min(), "Historical", "Forecast")

# ===============================
# 📈 Visualisasi
# ===============================
fig = go.Figure()

hist_df = combined_df[combined_df["Source"] == "Historical"]
fore_df = combined_df[combined_df["Source"] == "Forecast"]

fig.add_trace(go.Scatter(
    x=hist_df["Datetime"], y=hist_df["Water_level"],
    name="Water Level (Historis)", line=dict(color="blue")
))
fig.add_trace(go.Scatter(
    x=fore_df["Datetime"], y=fore_df["Water_level"],
    name="Water Level (Forecast 14 Hari)", line=dict(color="orange", dash="dot")
))
fig.add_vline(x=selected_date, line_width=2, line_dash="dash", line_color="red")

fig.update_layout(
    title="Prediksi Level Air Sungai (14 Hari ke Depan)",
    xaxis_title="Tanggal", yaxis_title="Level Air (m)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ===============================
# 💾 Unduh Hasil
# ===============================
csv = combined_df.to_csv(index=False).encode("utf-8")
st.download_button("💾 Unduh Hasil Prediksi (CSV)", csv, "water_level_forecast.csv", "text/csv")

st.success(f"✅ Prediksi selesai! Dimulai dari {selected_date.date()} hingga {selected_date.date() + timedelta(days=14)}.")

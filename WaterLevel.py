# ============================================
# WaterLevel_Tuhup_App.py
# ============================================
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

st.title("Water Level Prediction — Jetty Tuhup 🌊")

today = datetime.today().date()

# -----------------------------
# Pilihan tanggal prediksi
# -----------------------------
st.subheader("Pilih Tanggal Prediksi Water Level (Maks: H+1)")
pred_date = st.date_input(
    "Tanggal Prediksi Water Level",
    value=today + timedelta(days=1),
    min_value=today - timedelta(days=30),
    max_value=today + timedelta(days=1)
)

# -----------------------------
# Upload file CSV (AWLR Jetty Tuhup)
# -----------------------------
st.subheader("Upload Data Water Level (AWLR Jetty Tuhup)")
st.caption("Format: **Datetime | Level Air** (minimal 7 hari sebelum tanggal prediksi)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# -----------------------------
# Helper functions
# -----------------------------
def safe_get(df, date, col):
    try:
        return float(df.loc[date, col])
    except Exception:
        return 0.0

def fetch_archive(start_date, end_date):
    """Ambil data cuaca historis"""
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
    return df

def fetch_forecast():
    """Ambil data cuaca forecast (16 hari ke depan)"""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude=-0.61&longitude=114.8"
        f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
        f"&timezone=Asia%2FSingapore&forecast_days=16"
    )
    forecast = requests.get(url).json()
    df_forecast = pd.DataFrame({
        "time": forecast["daily"]["time"],
        "precipitation_sum": forecast["daily"]["precipitation_sum"],
        "temperature_mean": forecast["daily"]["temperature_2m_mean"],
        "relative_humidity": forecast["daily"]["relative_humidity_2m_mean"]
    })
    df_forecast["time"] = pd.to_datetime(df_forecast["time"]).dt.date
    df_forecast.set_index("time", inplace=True)
    return df_forecast

# -----------------------------
# Forecasting
# -----------------------------
if uploaded_file is not None:
    try:
        df_awlr = pd.read_csv(uploaded_file)
        df_awlr["Datetime"] = pd.to_datetime(df_awlr["Datetime"], errors="coerce")
        df_awlr = df_awlr.dropna(subset=["Datetime", "Level Air"])
        df_awlr["Date"] = df_awlr["Datetime"].dt.date
        df_awlr = df_awlr.groupby("Date", as_index=False)["Level Air"].mean()
        df_awlr = df_awlr.set_index("Date").sort_index()

        # Ambil 7 hari sebelum tanggal prediksi
        last_7 = df_awlr[df_awlr.index < pred_date].tail(7)

        if len(last_7) < 7:
            st.warning("⚠️ Data tidak mencukupi. Pastikan file berisi minimal 7 hari sebelum tanggal prediksi.")
        else:
            st.success("✅ Data valid. Siap untuk forecasting.")
            st.subheader("Preview Data Water Level (7 hari terakhir)")
            st.dataframe(last_7.style.format("{:.2f}"))

            if st.button("Mulai Forecast"):
                # -----------------------------
                # Siapkan fitur cuaca
                # -----------------------------
                start_hist = (pred_date - timedelta(days=7)).isoformat()
                end_hist = (pred_date - timedelta(days=1)).isoformat()
                df_hist = fetch_archive(start_hist, end_hist)

                # Gabungkan water level dari upload
                df_hist["water_level"] = None
                for d, v in last_7["Level Air"].items():
                    if d in df_hist.index:
                        df_hist.loc[d, "water_level"] = round(v, 2)

                df_hist = df_hist.fillna(0)

                # Jika tanggal prediksi di masa lalu → pakai kombinasi historis + forecast
                df_forecast = fetch_forecast()
                df = pd.concat([df_hist, df_forecast]).drop_duplicates().sort_index()

                # -----------------------------
                # Forecasting 7 hari ke depan
                # -----------------------------
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

                wl_lags = list(last_7["Level Air"].values)[::-1]  # 7 hari kebelakang
                wl_lags = wl_lags[::-1]  # dari lama ke baru

                results = {}
                for step in range(1, 8):
                    pred_day = pred_date + timedelta(days=step)
                    inp = {}

                    for i in range(1, 8):
                        date_i = pred_day - timedelta(days=i)
                        inp[f"Precipitation_lag{i}d"] = [safe_get(df, date_i, "precipitation_sum")]
                    for i in range(1, 5):
                        date_i = pred_day - timedelta(days=i)
                        inp[f"Temperature_lag{i}d"] = [safe_get(df, date_i, "temperature_mean")]
                    for i in range(1, 8):
                        date_i = pred_day - timedelta(days=i)
                        inp[f"Relative_humidity_lag{i}d"] = [safe_get(df, date_i, "relative_humidity")]
                    for i in range(1, 8):
                        inp[f"Water_level_lag{i}d"] = [wl_lags[-i]]

                    input_df = pd.DataFrame(inp)[features].fillna(0)
                    pred = model.predict(input_df)[0]
                    results[pred_day] = pred
                    wl_lags.append(pred)
                    wl_lags = wl_lags[-7:]

                # -----------------------------
                # Tampilkan hasil
                # -----------------------------
                st.success("✅ Forecast selesai!")
                df_result = pd.DataFrame({
                    "Tanggal": list(results.keys()),
                    "Prediksi Water Level (m)": list(results.values())
                })
                st.dataframe(df_result.style.format("{:.2f}"))

                # Plot hasil
                fig = go.Figure()
                # Historis (dari file upload)
                fig.add_trace(go.Scatter(
                    x=last_7.index, y=last_7["Level Air"],
                    mode="lines+markers", name="Historical", line=dict(color="blue")
                ))
                # Forecast
                fig.add_trace(go.Scatter(
                    x=df_result["Tanggal"], y=df_result["Prediksi Water Level (m)"],
                    mode="lines+markers", name="Forecast", line=dict(color="green", dash="dash")
                ))

                # Batas aman
                lower_limit, upper_limit = 19.5, 26.5
                fig.add_hline(y=lower_limit, line=dict(color="red", width=2, dash="dash"),
                              annotation_text="Lower Limit", annotation_position="bottom left")
                fig.add_hline(y=upper_limit, line=dict(color="red", width=2, dash="dash"),
                              annotation_text="Upper Limit", annotation_position="top left")

                fig.update_layout(
                    title=f"Water Level Forecast — Jetty Tuhup ({pred_date.strftime('%d %B %Y')})",
                    xaxis_title="Tanggal",
                    yaxis_title="Water Level (m)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")

# ============================================
# WaterLevel_JettyTuhup_App.py
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

st.subheader("Pilih Tanggal Prediksi (maksimal hari ini)")
pred_date = st.date_input(
    "Tanggal Prediksi Water Level",
    value=today,
    max_value=today
)

st.caption("📤 Upload data AWLR Jetty Tuhup (2 kolom: Datetime | Level Air, minimal 7 hari terakhir)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ------------------------------------------
# Fungsi bantu
# ------------------------------------------
def safe_get(df, date, col):
    try:
        return float(df.loc[date, col])
    except Exception:
        return 0.0

def fetch_meteo_data(start_date, end_date):
    """Ambil data Open-Meteo (historis)"""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=-0.61&longitude=114.8"
        f"&start_date={start_date}&end_date={end_date}"
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

def fetch_forecast_data(days_ahead):
    """Ambil data forecast hingga n hari ke depan"""
    url_forecast = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude=-0.61&longitude=114.8"
        f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
        f"&timezone=Asia%2FSingapore&forecast_days=16"
    )
    forecast = requests.get(url_forecast).json()
    df_forecast = pd.DataFrame({
        "time": forecast["daily"]["time"],
        "precipitation_sum": forecast["daily"]["precipitation_sum"],
        "temperature_mean": forecast["daily"]["temperature_2m_mean"],
        "relative_humidity": forecast["daily"]["relative_humidity_2m_mean"]
    })
    df_forecast["time"] = pd.to_datetime(df_forecast["time"]).dt.date
    df_forecast = df_forecast.iloc[1:days_ahead+1]
    df_forecast.set_index("time", inplace=True)
    return df_forecast


# ------------------------------------------
# Jalankan prediksi
# ------------------------------------------
if uploaded_file is not None:
    try:
        df_awlr = pd.read_csv(uploaded_file)
        df_awlr["Datetime"] = pd.to_datetime(df_awlr["Datetime"])
        df_awlr["Date"] = df_awlr["Datetime"].dt.date
        df_awlr = df_awlr.groupby("Date", as_index=False)["Level Air"].mean()
        df_awlr = df_awlr.set_index("Date").sort_index()

        st.write("### Preview Data Water Level")
        st.dataframe(df_awlr.tail(7))

        # ambil 7 hari terakhir sebelum tanggal prediksi
        last_7 = df_awlr[df_awlr.index < pred_date].tail(7)
        if len(last_7) < 7:
            st.warning("⚠️ Data tidak mencukupi. Pastikan file mencakup minimal 7 hari sebelum tanggal prediksi.")
        else:
            st.success("✅ Data valid. Siap prediksi.")
            if st.button("Predict Water Level"):
                features = [
                    "Precipitation_lag1d","Precipitation_lag2d","Precipitation_lag3d",
                    "Precipitation_lag4d","Precipitation_lag5d","Precipitation_lag6d","Precipitation_lag7d",
                    "Temperature_lag1d","Temperature_lag2d","Temperature_lag3d","Temperature_lag4d",
                    "Relative_humidity_lag1d","Relative_humidity_lag2d","Relative_humidity_lag3d",
                    "Relative_humidity_lag4d","Relative_humidity_lag5d","Relative_humidity_lag6d",
                    "Relative_humidity_lag7d",
                    "Water_level_lag1d","Water_level_lag2d","Water_level_lag3d","Water_level_lag4d",
                    "Water_level_lag5d","Water_level_lag6d","Water_level_lag7d",
                ]

                # ambil data historis 7 hari sebelum tanggal prediksi
                start_hist = (pred_date - timedelta(days=7)).isoformat()
                end_hist = (pred_date - timedelta(days=1)).isoformat()
                df_hist = fetch_meteo_data(start_hist, end_hist)

                # gabungkan dengan level air
                df_hist["water_level"] = None
                for d, v in last_7["Level Air"].items():
                    if d in df_hist.index:
                        df_hist.loc[d, "water_level"] = round(v, 2)

                df_hist = df_hist.fillna(0)

                # cek apakah user minta tanggal lampau
                if pred_date < today:
                    # prediksi gabungan historis + forecast validasi
                    n_future = (today - pred_date).days
                    df_future = fetch_forecast_data(n_future)
                    df = pd.concat([df_hist, df_future]).drop_duplicates().sort_index()
                else:
                    df = df_hist.copy()

                # autoregresi 7 hari ke depan
                results = {}
                wl_lags = list(last_7["Level Air"].iloc[::-1])[:7]  # 7 hari ke belakang
                wl_lags = wl_lags[::-1]  # urut dari lama ke baru

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

                st.success("✅ Prediksi selesai!")
                result_df = pd.DataFrame({
                    "Date": list(results.keys()),
                    "Predicted Water Level (m)": list(results.values())
                })
                st.dataframe(result_df.style.format("{:.2f}"))

                # Visualisasi
                df_plot_hist = df_awlr.reset_index()[["Date", "Level Air"]]
                df_plot_pred = result_df.rename(columns={"Predicted Water Level (m)": "Level Air"})

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_plot_hist["Date"], y=df_plot_hist["Level Air"],
                    mode="lines+markers", name="Historical", line=dict(color="blue")
                ))
                fig.add_trace(go.Scatter(
                    x=df_plot_pred["Date"], y=df_plot_pred["Level Air"],
                    mode="lines+markers", name="Forecast", line=dict(color="green", dash="dash")
                ))
                fig.update_layout(
                    title="Water Level Forecast — Jetty Tuhup",
                    xaxis_title="Date", yaxis_title="Water Level (m)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error membaca file: {e}")

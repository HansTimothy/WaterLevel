# WaterLevel_API_fixed.py
import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊")

today = datetime.today().date()

st.subheader("Pilih Tanggal Awal Prediksi Water Level")

pred_date = st.date_input(
    "Tanggal Mulai Prediksi",
    value=today,
    min_value=date(2018, 5, 8),
    max_value=today
)

# -----------------------------
# Input manual Water Level
# -----------------------------
st.subheader("Masukkan Data Water Level (7 hari terakhir)")

# Input manual dari hari ini ke belakang 6 hari
wl_dates = [today - timedelta(days=i) for i in range(0, 7)]
wl_inputs = [
    st.number_input(f"Water Level **{d.strftime('%d %B %Y')}**", value=20.0, format="%.2f")
    for d in wl_dates
]

# -----------------------------
# Helper function
# -----------------------------
def safe_get(df, date, col):
    try:
        return float(df.loc[date, col])
    except Exception:
        return 0.0


# -----------------------------
# Fetch data & predict
# -----------------------------
if st.button("Fetch Data & Predict 14 Hari ke Depan"):

    # Ambil archive data 7 hari sebelum pred_date sampai hari ini
    start_date = (pred_date - timedelta(days=7)).isoformat()
    end_date = today.isoformat()

    url_archive = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=-0.61&longitude=114.8&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
        f"&timezone=Asia%2FSingapore"
    )

    archive = requests.get(url_archive).json()

    df_archive = pd.DataFrame({
        "time": archive["daily"]["time"],
        "precipitation_sum": archive["daily"]["precipitation_sum"],
        "temperature_mean": archive["daily"]["temperature_2m_mean"],
        "relative_humidity": archive["daily"]["relative_humidity_2m_mean"]
    })
    df_archive["time"] = pd.to_datetime(df_archive["time"]).dt.date

    # Ambil forecast data untuk 14 hari ke depan
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

    # Gabungkan archive + forecast
    df = pd.concat([df_archive, df_forecast]).drop_duplicates().sort_values("time")
    df.set_index("time", inplace=True)
    df["water_level"] = None

    # Isi data water level manual (hari ini dan 6 hari ke belakang)
    for i, d in enumerate(wl_dates):
        if d in df.index:
            df.loc[d, "water_level"] = round(float(wl_inputs[i]), 2)

    # -----------------------------
    # Prediksi iteratif 14 hari ke depan
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

    # Lags awal dari input manual (H0 terbaru di index 0)
    water_level_lags = wl_inputs[:]
    forecast_dates = []
    results = {}

    for step in range(1, 15):  # prediksi 14 hari
        pred_day = today + timedelta(days=step)

        inp = {}
        # Cuaca lags
        for i in range(1, 8):
            date_i = pred_day - timedelta(days=i)
            inp[f"Precipitation_lag{i}d"] = [safe_get(df, date_i, "precipitation_sum")]
        for i in range(1, 5):
            date_i = pred_day - timedelta(days=i)
            inp[f"Temperature_lag{i}d"] = [safe_get(df, date_i, "temperature_mean")]
        for i in range(1, 8):
            date_i = pred_day - timedelta(days=i)
            inp[f"Relative_humidity_lag{i}d"] = [safe_get(df, date_i, "relative_humidity")]

        # Water level lags
        for i in range(1, 8):
            inp[f"Water_level_lag{i}d"] = [water_level_lags[i-1]]

        input_data = pd.DataFrame(inp)[features].fillna(0.0)
        prediction = model.predict(input_data)[0]
        results[pred_day] = round(prediction, 2)

        water_level_lags = [prediction] + water_level_lags[:-1]
        df.loc[pred_day, "water_level"] = round(prediction, 2)
        forecast_dates.append(pred_day)

    # Pastikan kolom Date benar dan dalam format date
    df_preview["Date"] = pd.to_datetime(df_preview["Date"], errors="coerce").dt.date
    
    # Kolom numerik yang mau diformat
    numeric_cols = df_preview.select_dtypes(include=["float64", "int64"]).columns
    
    # Fungsi highlight area forecast (prediksi)
    def highlight_forecast(row):
        # Jika tanggal melebihi hari ini = data prediksi
        if row["Date"] > today:
            # beri warna biru muda transparan
            return ["background-color: rgba(135, 206, 250, 0.3)"] * len(row)
        else:
            # data historis tanpa warna
            return [""] * len(row)
    
    # Tampilkan di Streamlit
    st.subheader("📊 Data Preview")
    
    st.dataframe(
        df_preview.style
            .apply(highlight_forecast, axis=1)
            .format("{:.2f}", subset=numeric_cols)
            .set_properties(**{"text-align": "right"}, subset=numeric_cols)
    )

    # -----------------------------
    # Plot
    # -----------------------------
    df_plot = df.reset_index()[["time", "water_level"]].rename(columns={"time": "Date"})
    lower_limit = 19.5
    upper_limit = 26.5

    df_hist = df_plot[df_plot["Date"] <= today]
    df_pred = df_plot[df_plot["Date"] > today]

    df_pred_safe = df_pred[df_pred["water_level"].between(lower_limit, upper_limit)]
    df_pred_unsafe = df_pred[(df_pred["water_level"] < lower_limit) | (df_pred["water_level"] > upper_limit)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_hist["Date"],
        y=df_hist["water_level"],
        mode="lines+markers",
        line=dict(color="blue", width=2),
        marker=dict(color="blue", size=8),
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=df_pred_safe["Date"],
        y=df_pred_safe["water_level"],
        mode="lines+markers",
        line=dict(color="black", width=2, dash="dash"),
        marker=dict(color="green", size=8),
        name="Prediction (Loadable)"
    ))

    fig.add_trace(go.Scatter(
        x=df_pred_unsafe["Date"],
        y=df_pred_unsafe["water_level"],
        mode="lines+markers",
        line=dict(color="black", width=2, dash="dash"),
        marker=dict(color="red", size=8),
        name="Prediction (Unloadable)"
    ))

    fig.add_hline(y=lower_limit, line=dict(color="red", width=2, dash="dash"),
                  annotation_text="Lower Limit", annotation_position="bottom left")
    fig.add_hline(y=upper_limit, line=dict(color="red", width=2, dash="dash"),
                  annotation_text="Upper Limit", annotation_position="top left")

    # Tambahkan area RMSE
    rmse = 0.87
    if not df_pred.empty:
        upper_band = df_pred["water_level"] + rmse
        lower_band = df_pred["water_level"] - rmse
        fig.add_trace(go.Scatter(
            x=df_pred["Date"],
            y=lower_band,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df_pred["Date"],
            y=upper_band,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.1)",
            name="Prediction error ±0.87 m"
        ))

    tick_text = [d.strftime("%d/%m/%y") for d in df_plot["Date"]]
    fig.update_layout(
        title=f"Prediksi Water Level {pred_date.strftime('%d %B %Y')} s.d. {pred_date + timedelta(days=14):%d %B %Y}",
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        xaxis=dict(tickangle=90, tickvals=df_plot["Date"], ticktext=tick_text),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

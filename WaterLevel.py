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

st.subheader("Pilih Tanggal Prediksi Water Level (Max: H+14)")

pred_date = st.date_input(
    "Tanggal Prediksi Water Level",
    value=today + timedelta(days=1),
    min_value=date(2018, 5, 8),
    max_value=today + timedelta(days=14)
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
st.subheader("Masukkan Data Water Level")

if pred_date <= today + timedelta(days=1):
    # Prediksi untuk masa lalu atau H+1:
    wl_dates = [pred_date - timedelta(days=i) for i in range(1, 8)]
else:
    # Prediksi H+2..H+7:
    wl_dates = [today - timedelta(days=i) for i in range(0, 7)]

wl_inputs = [
    st.number_input(f"Water Level **{d.strftime('%d %B %Y')}**", value=20.0, format="%.2f")
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

    if pred_date <= today + timedelta(days=1):
        # -----------------------------
        # Skenario 1: prediksi masa lalu / H+1 (pakai archive)
        # -----------------------------
        start_date = (pred_date - timedelta(days=7)).isoformat()
        end_date = (pred_date - timedelta(days=1)).isoformat()

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

        # Tambah kolom water level manual
        for i, d in enumerate(wl_dates):
            if d in df.index:
                df.loc[d, "water_level"] = round(float(wl_inputs[i]), 2)

        df_preview = df.copy()
        if "water_level" in df_preview.columns:
            wl = df_preview.pop("water_level")
            df_preview.insert(0, "water_level", wl)
        
        numeric_cols = ["precipitation_sum", "temperature_mean", "water_level"]
        
        st.subheader("Preview Data Historis")
        st.dataframe(df_preview.style.format("{:.2f}", subset=numeric_cols).set_properties(**{"text-align": "right"}, subset=numeric_cols))

        # buat input feature
        inp = {}
        for i in range(1, 8):
            date_i = pred_date - timedelta(days=i)
            inp[f"Precipitation_lag{i}d"] = [safe_get(df, date_i, "precipitation_sum")]
        for i in range(1, 5):
            date_i = pred_date - timedelta(days=i)
            inp[f"Temperature_lag{i}d"] = [safe_get(df, date_i, "temperature_mean")]
        for i in range(1, 8):
            date_i = pred_date - timedelta(days=i)
            inp[f"Relative_humidity_lag{i}d"] = [safe_get(df, date_i, "relative_humidity")]
        for i in range(1, 8):
            inp[f"Water_level_lag{i}d"] = [wl_inputs[i-1]]

        input_data = pd.DataFrame(inp)[features].fillna(0.0)

        prediction = model.predict(input_data)[0]
        df.loc[pred_date, "water_level"] = round(prediction, 2)
        st.success(f"Predicted Water Level on {pred_date.strftime('%d %B %Y')}: {prediction:.2f} m")

    else:
        # -----------------------------
        # Skenario 2: prediksi H+2..H+14 (pakai archive + forecast)
        # -----------------------------
        start_hist = (today - timedelta(days=6)).isoformat()
        end_hist = today.isoformat()
        url_hist = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=-0.61&longitude=114.8&start_date={start_hist}&end_date={end_hist}"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        hist = requests.get(url_hist).json()
        df_hist = pd.DataFrame({
            "time": hist["daily"]["time"],
            "precipitation_sum": hist["daily"]["precipitation_sum"],
            "temperature_mean": hist["daily"]["temperature_2m_mean"],
            "relative_humidity": hist["daily"]["relative_humidity_2m_mean"]
        })
        df_hist["time"] = pd.to_datetime(df_hist["time"]).dt.date
        
        n_days = (pred_date - today).days  
        
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
        df_forecast = df_forecast.iloc[1:n_days+1]
        df_forecast["time"] = pd.to_datetime(df_forecast["time"]).dt.date
        
        df = pd.concat([df_hist, df_forecast]).drop_duplicates().sort_values("time")
        df.set_index("time", inplace=True)
        df["water_level"] = None
        
        for i, d in enumerate(wl_dates):
            if d in df.index:
                df.loc[d, "water_level"] = round(float(wl_inputs[i]), 2)
        
        water_level_lags = wl_inputs[:]  
        results = {}
        forecast_dates = []  
        
        for step in range(1, n_days + 1):
            pred_day = today + timedelta(days=step)
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
                inp[f"Water_level_lag{i}d"] = [water_level_lags[i-1]]
        
            input_data = pd.DataFrame(inp)[features].fillna(0.0)
            prediction = model.predict(input_data)[0]
            results[pred_day] = prediction
            water_level_lags = [prediction] + water_level_lags[:-1]
            if pred_day in df.index:
                df.loc[pred_day, "water_level"] = round(prediction, 2)
                forecast_dates.append(pred_day)
        
        df_preview = df.copy()
        if "water_level" in df_preview.columns:
            wl = df_preview.pop("water_level")
            df_preview.insert(0, "water_level", wl)
        
        df_preview = df_preview.iloc[:-1]
        def highlight_forecast(row):
            return ['background-color: #bce4f6' if row.name in forecast_dates else '' for _ in row]
        
        numeric_cols = ["precipitation_sum", "temperature_mean", "water_level"]
        st.subheader("Preview Data (History + Forecast)")
        st.dataframe(df_preview.style.apply(highlight_forecast, axis=1).format("{:.2f}", subset=numeric_cols).set_properties(**{"text-align": "right"}, subset=numeric_cols))
        
        last_date, last_val = list(results.items())[-1]
        st.subheader("Hasil Prediksi")
        st.success(f"Predicted Water Level on {last_date.strftime('%d %B %Y')}: {last_val:.2f} m")

    # -----------------------------
    # PLOT SECTION (fixed for all scenarios)
    # -----------------------------
    df_plot = df.reset_index()[["time", "water_level"]].copy()
    df_plot.rename(columns={"time": "Date"}, inplace=True)

    lower_limit = 19.5
    upper_limit = 26.5
    today = datetime.today().date()

    # --- Tentukan rentang histori & prediksi ---
    if pred_date <= today + timedelta(days=1):
        df_hist = df_plot[df_plot["Date"] <= pred_date]
        df_pred = df_plot[df_plot["Date"] >= pred_date]
    else:
        df_hist = df_plot[df_plot["Date"] <= today]
        df_pred = df_plot[df_plot["Date"] >= today]

    df_pred_safe = df_pred[df_pred["water_level"].between(lower_limit, upper_limit)]
    df_pred_unsafe = df_pred[(df_pred["water_level"] < lower_limit) | (df_pred["water_level"] > upper_limit)]

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=df_hist["Date"],
        y=df_hist["water_level"],
        mode="lines+markers",
        line=dict(color="blue", width=2),
        marker=dict(color="blue", size=8),
        name="Historical"
    ))

    # Dashed prediction base
    fig.add_trace(go.Scatter(
        x=df_pred["Date"],
        y=df_pred["water_level"],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        showlegend=False,
    ))

    # Loadable predictions
    fig.add_trace(go.Scatter(
        x=df_pred_safe["Date"],
        y=df_pred_safe["water_level"],
        mode="lines+markers",
        line=dict(color="black", width=2, dash="dash"),
        marker=dict(color="green", size=8),
        name="Prediction (Loadable)"
    ))

    # Unloadable predictions
    fig.add_trace(go.Scatter(
        x=df_pred_unsafe["Date"],
        y=df_pred_unsafe["water_level"],
        mode="lines+markers",
        line=dict(color="black", width=2, dash="dash"),
        marker=dict(color="red", size=8),
        name="Prediction (Unloadable)"
    ))

    # Today marker
    today_point = df_plot[df_plot["Date"] == today]
    if not today_point.empty:
        fig.add_trace(go.Scatter(
            x=today_point["Date"],
            y=today_point["water_level"],
            mode="markers",
            marker=dict(color="blue", size=8, symbol="circle"),
            name="Today",
            showlegend=False
        ))

    # Loadable limits
    fig.add_hline(y=lower_limit, line=dict(color="red", width=2, dash="dash"),
                  annotation_text="Lower Limit", annotation_position="bottom left")
    fig.add_hline(y=upper_limit, line=dict(color="red", width=2, dash="dash"),
                  annotation_text="Upper Limit", annotation_position="top left")

    # ±RMSE area
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
            name="Prediction error ±0.87 m",
            showlegend=True
        ))

    all_dates = df_plot["Date"]
    tick_text = [d.strftime("%d/%m/%y") for d in all_dates]

    fig.update_layout(
        title="Water Level Dashboard 🌊",
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        xaxis=dict(
            tickangle=90,
            tickmode="array",
            tickvals=all_dates,
            ticktext=tick_text
        ),
        yaxis=dict(autorange=True),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

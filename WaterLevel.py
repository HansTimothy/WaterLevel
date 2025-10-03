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

st.subheader("Pilih Tanggal Prediksi Water Level (Max: H+16)")

pred_date = st.date_input(
    "Tanggal Prediksi Water Level",
    value=today + timedelta(days=1),
    max_value=today + timedelta(days=16)
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
    # kita membutuhkan water level untuk pred_date-1 .. pred_date-7
    wl_dates = [pred_date - timedelta(days=i) for i in range(1, 8)]
else:
    # Prediksi H+2..H+7:
    # user mau memasukkan water level dari hari ini (H0) sampai H-6
    # sehingga wl_inputs[0] = hari ini (paling baru)
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
        # Skenario 1: prediksi masa lalu / H+1 (pakai archive saja)
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
        # Isi dari input manual (H0..H-6)
        for i, d in enumerate(wl_dates):
            if d in df.index:
                df.loc[d, "water_level"] = round(float(wl_inputs[i]), 2)

        # Setelah loop prediksi selesai dan water_level sudah diisi
        df_preview = df.copy()
        
        if "water_level" in df_preview.columns:
            wl = df_preview.pop("water_level")
            df_preview.insert(0, "water_level", wl)
        
        # List kolom numeric untuk formatting
        numeric_cols = ["precipitation_sum", "temperature_mean", "water_level"]
        
        st.subheader("Preview Data Historis")
        st.dataframe(df_preview.style.format("{:.2f}", subset=numeric_cols).set_properties(**{"text-align": "right"}, subset=numeric_cols))

        # buat input feature — gunakan safe_get untuk menghindari missing
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
        # WATER LEVEL: wl_inputs di-build sedemikian sehingga wl_inputs[0] == pred_date-1
        for i in range(1, 8):
            inp[f"Water_level_lag{i}d"] = [wl_inputs[i-1]]

        input_data = pd.DataFrame(inp)[features].fillna(0.0)

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Water Level on {pred_date.strftime('%d %B %Y')}: {prediction:.2f} m")

    else:
        # -----------------------------
        # Skenario 2: prediksi H+2..H+7 (pakai archive + forecast, looping)
        # -----------------------------
        # Ambil historis H-6 .. H0
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
        
        n_days = (pred_date - today).days  # misal H+2 -> n_days=2
        
        # Ambil forecast (H+0..H+7)
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
        # ambil hanya sampai n_days (H+1..H+n)
        df_forecast = df_forecast.iloc[1:n_days+1]
        df_forecast["time"] = pd.to_datetime(df_forecast["time"]).dt.date
        
        # gabungkan histori + forecast
        df = pd.concat([df_hist, df_forecast]).drop_duplicates().sort_values("time")
        df.set_index("time", inplace=True)
        
        # Tambah kolom water level
        df["water_level"] = None
        
        # Isi dari input manual (H0..H-6)
        for i, d in enumerate(wl_dates):
            if d in df.index:
                df.loc[d, "water_level"] = round(float(wl_inputs[i]), 2)
        
        # Siapkan lags untuk water level (H0 terbaru -> index 0)
        water_level_lags = wl_inputs[:]  # [H0, H-1, H-2, ..., H-6]
        
        results = {}
        forecast_dates = []  # untuk highlight
        
        for step in range(1, n_days + 1):
            pred_day = today + timedelta(days=step)
        
            inp = {}
            # lags cuaca
            for i in range(1, 8):
                date_i = pred_day - timedelta(days=i)
                inp[f"Precipitation_lag{i}d"] = [safe_get(df, date_i, "precipitation_sum")]
            for i in range(1, 5):
                date_i = pred_day - timedelta(days=i)
                inp[f"Temperature_lag{i}d"] = [safe_get(df, date_i, "temperature_mean")]
            for i in range(1, 8):
                date_i = pred_day - timedelta(days=i)
                inp[f"Relative_humidity_lag{i}d"] = [safe_get(df, date_i, "relative_humidity")]
        
            # lags water level
            for i in range(1, 8):
                inp[f"Water_level_lag{i}d"] = [water_level_lags[i-1]]
        
            # buat dataframe input
            input_data = pd.DataFrame(inp)[features].fillna(0.0)
        
            # prediksi
            prediction = model.predict(input_data)[0]
            results[pred_day] = prediction
        
            # update lags
            water_level_lags = [prediction] + water_level_lags[:-1]
        
            # update df (biar preview langsung terlihat)
            if pred_day in df.index:
                df.loc[pred_day, "water_level"] = round(prediction, 2)
                forecast_dates.append(pred_day)
        
        # --- letakkan water_level di kolom kedua ---
        df_preview = df.copy()
        if "water_level" in df_preview.columns:
            wl = df_preview.pop("water_level")
            df_preview.insert(0, "water_level", wl)  # kolom pertama

        df_preview = df_preview.iloc[:-1]
        
        # styling untuk highlight forecast
        def highlight_forecast(row):
            return ['background-color: #bce4f6' if row.name in forecast_dates else '' for _ in row]
        
        numeric_cols = ["precipitation_sum", "temperature_mean", "water_level"]
        
        st.subheader("Preview Data (History + Forecast)")
        st.dataframe(df_preview.style.apply(highlight_forecast, axis=1).format("{:.2f}", subset=numeric_cols).set_properties(**{"text-align": "right"}, subset=numeric_cols))
        
        # Ambil hasil prediksi terakhir
        last_date, last_val = list(results.items())[-1]
        st.subheader("Hasil Prediksi")
        st.success(f"Predicted Water Level on {last_date.strftime('%d %B %Y')}: {last_val:.2f} m")

    # Ambil kolom yang diperlukan
    # Ambil kolom yang diperlukan
    df_plot = df.reset_index()[["time", "water_level"]].copy()
    df_plot.rename(columns={"time": "Date"}, inplace=True)
    
    # Tentukan warna: merah jika non-sailable, biru jika normal
    lower_limit = 19.5
    upper_limit = 26.5
    df_plot["color"] = df_plot["water_level"].apply(lambda x: "red" if (x < lower_limit or x > upper_limit) else "blue")
    
    # Highlight tanggal prediksi terakhir
    pred_date = df_plot["Date"].max()  # asumsikan prediksi terakhir ada di baris terakhir
    df_plot["marker_size"] = df_plot["Date"].apply(lambda x: 15 if x == pred_date else 10)
    df_plot["marker_color"] = df_plot["Date"].apply(lambda x: "green" if x == pred_date else df_plot.loc[df_plot["Date"] == x, "color"].values[0])
    
    # Buat figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_plot["Date"],
        y=df_plot["water_level"],
        mode="lines+markers",
        marker=dict(color=df_plot["marker_color"], size=df_plot["marker_size"]),
        line=dict(color="blue", width=2),
        name="Water Level"
    ))
    
    # Tambahkan batas atas/bawah
    fig.add_hline(y=lower_limit, line=dict(color="orange", width=2, dash="dash"),
                  annotation_text="Lower Limit", annotation_position="bottom left")
    fig.add_hline(y=upper_limit, line=dict(color="orange", width=2, dash="dash"),
                  annotation_text="Upper Limit", annotation_position="top left")
    
    fig.update_layout(
        title="Water Level Dashboard 🌊",
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[lower_limit-1, upper_limit+1]),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

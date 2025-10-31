# WaterLevel_JettyTuhup.py
import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction - Jetty Tuhup 🌊")

st.markdown("""
**Instruksi Upload File AWLR Jetty Tuhup:**  
- Format CSV, kolom minimal: `Datetime`, `Water_level`  
- Data terakhir wajib mencakup **7 hari terakhir** untuk forecast autoregresi
""")

# -----------------------------
# Pilih tanggal prediksi
# -----------------------------
today = datetime.today().date()

pred_date = st.date_input(
    "Pilih Tanggal Prediksi (Max H+1)",
    value=today,
    max_value=today + timedelta(days=1)
)

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV AWLR Jetty Tuhup", type=["csv"])

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        df_upload['Datetime'] = pd.to_datetime(df_upload['Datetime']).dt.date
        df_upload.set_index('Datetime', inplace=True)
        st.success("File berhasil di-upload!")
        
        # Ambil 7 hari terakhir water level untuk input lag
        wl_inputs = []
        wl_dates = [pred_date - timedelta(days=i) for i in range(1, 8)]
        for d in wl_dates:
            wl = df_upload.loc[d, 'Water_level'] if d in df_upload.index else None
            wl_inputs.append(float(wl) if wl is not None else 0.0)
        
        st.subheader("Preview 7 Hari Terakhir Water Level")
        df_preview = pd.DataFrame({
            'Date': wl_dates[::-1],
            'Water_level': wl_inputs[::-1]
        })
        st.dataframe(df_preview.style.format("{:.2f}", subset=['Water_level']).set_properties(**{"text-align": "right"}, subset=['Water_level']))
        
        # -----------------------------
        # Tombol Forecast
        # -----------------------------
        if st.button("Forecast 14 Hari ke Depan"):
            
            progress_text = "Forecast sedang berjalan..."
            my_bar = st.progress(0, text=progress_text)
            
            # -----------------------------
            # Ambil cuaca: 7 hari historis + 14 hari forecast
            # -----------------------------
            start_hist = (pred_date - timedelta(days=7)).isoformat()
            end_hist = (pred_date - timedelta(days=1)).isoformat()
            url_hist = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude=-0.61&longitude=114.8&start_date={start_hist}&end_date={end_hist}"
                f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
                f"&timezone=Asia%2FSingapore"
            )
            hist = requests.get(url_hist).json()
            df_hist = pd.DataFrame({
                "Date": pd.to_datetime(hist["daily"]["time"]).date,
                "Precipitation": hist["daily"]["precipitation_sum"],
                "Temperature": hist["daily"]["temperature_2m_mean"],
                "Relative_humidity": hist["daily"]["relative_humidity_2m_mean"]
            })
            df_hist.set_index("Date", inplace=True)
            
            # Forecast cuaca 14 hari
            url_forecast = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude=-0.61&longitude=114.8"
                f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
                f"&timezone=Asia%2FSingapore&forecast_days=16"
            )
            forecast = requests.get(url_forecast).json()
            df_forecast = pd.DataFrame({
                "Date": pd.to_datetime(forecast["daily"]["time"]).date,
                "Precipitation": forecast["daily"]["precipitation_sum"],
                "Temperature": forecast["daily"]["temperature_2m_mean"],
                "Relative_humidity": forecast["daily"]["relative_humidity_2m_mean"]
            })
            df_forecast.set_index("Date", inplace=True)
            df_forecast = df_forecast.loc[pred_date:pred_date+timedelta(days=13)]
            
            # Gabungkan historical + forecast
            df_all = pd.concat([df_hist, df_forecast])
            
            # Tambah water_level kolom untuk autoregresi
            df_all["Water_level"] = None
            water_level_lags = wl_inputs[::-1]  # H-7 -> H-1
            results = {}
            forecast_dates = []
            
            for i, day in enumerate(df_forecast.index):
                inp = {}
                # cuaca lag
                for j in range(1, 8):
                    date_j = day - timedelta(days=j)
                    inp[f"Precipitation_lag{j}d"] = [df_all.loc[date_j, "Precipitation"] if date_j in df_all.index else 0.0]
                for j in range(1, 5):
                    date_j = day - timedelta(days=j)
                    inp[f"Temperature_lag{j}d"] = [df_all.loc[date_j, "Temperature"] if date_j in df_all.index else 0.0]
                for j in range(1, 8):
                    date_j = day - timedelta(days=j)
                    inp[f"Relative_humidity_lag{j}d"] = [df_all.loc[date_j, "Relative_humidity"] if date_j in df_all.index else 0.0]
                
                # water level lag
                for j in range(1, 8):
                    inp[f"Water_level_lag{j}d"] = [water_level_lags[j-1]]
                
                input_data = pd.DataFrame(inp)[[
                    f"Precipitation_lag{i}d" for i in range(1,8)] +
                    [f"Temperature_lag{i}d" for i in range(1,5)] +
                    [f"Relative_humidity_lag{i}d" for i in range(1,8)] +
                    [f"Water_level_lag{i}d" for i in range(1,8)]
                ].fillna(0.0)
                
                # predict
                pred_wl = model.predict(input_data)[0]
                results[day] = pred_wl
                df_all.loc[day, "Water_level"] = round(pred_wl, 2)
                
                # update lag
                water_level_lags = [pred_wl] + water_level_lags[:-1]
                
                # update progress bar
                my_bar.progress((i+1)/14, text=progress_text)
                time.sleep(0.05)
            
            # Preview final historical + forecast
            st.subheader("Preview Historical + Forecast")
            def highlight_forecast(row):
                return ['background-color: #bce4f6' if row.name in df_forecast.index else '' for _ in row]
            
            st.dataframe(df_all.style.apply(highlight_forecast, axis=1)
                         .format("{:.2f}", subset=["Water_level", "Precipitation", "Temperature", "Relative_humidity"])
                         .set_properties(**{"text-align": "right"}, subset=["Water_level", "Precipitation", "Temperature", "Relative_humidity"]))
            
            st.subheader("Hasil Forecast 14 Hari ke Depan")
            last_date, last_val = list(results.items())[-1]
            st.success(f"Prediksi Water Level terakhir: {last_val:.2f} m pada {last_date.strftime('%d %B %Y')}")
            
            # -----------------------------
            # Plot
            # -----------------------------
            lower_limit, upper_limit = 19.5, 28
            df_plot = df_all.reset_index()
            
            df_hist_plot = df_plot[df_plot["Date"] < pred_date]
            df_fore_plot = df_plot[df_plot["Date"] >= pred_date]
            df_fore_safe = df_fore_plot[df_fore_plot["Water_level"].between(lower_limit, upper_limit)]
            df_fore_unsafe = df_fore_plot[(df_fore_plot["Water_level"] < lower_limit) | (df_fore_plot["Water_level"] > upper_limit)]
            
            fig = go.Figure()
            # Historical
            fig.add_trace(go.Scatter(x=df_hist_plot["Date"], y=df_hist_plot["Water_level"],
                                     mode="lines+markers", line=dict(color="blue", width=2),
                                     marker=dict(color="blue", size=8), name="Historical"))
            # Forecast line
            fig.add_trace(go.Scatter(x=df_fore_plot["Date"], y=df_fore_plot["Water_level"],
                                     mode="lines", line=dict(color="black", width=2, dash="dash"), showlegend=False))
            # Safe forecast
            fig.add_trace(go.Scatter(x=df_fore_safe["Date"], y=df_fore_safe["Water_level"],
                                     mode="lines+markers", line=dict(color="black", width=2, dash="dash"),
                                     marker=dict(color="green", size=8), name="Forecast (Safe)"))
            # Unsafe forecast
            fig.add_trace(go.Scatter(x=df_fore_unsafe["Date"], y=df_fore_unsafe["Water_level"],
                                     mode="lines+markers", line=dict(color="black", width=2, dash="dash"),
                                     marker=dict(color="red", size=8), name="Forecast (Unsafe)"))
            
            # batas loadable
            fig.add_hline(y=lower_limit, line=dict(color="red", width=2, dash="dash"), annotation_text="Lower Limit", annotation_position="bottom left")
            fig.add_hline(y=upper_limit, line=dict(color="red", width=2, dash="dash"), annotation_text="Upper Limit", annotation_position="top left")
            
            fig.update_layout(title="Water Level Dashboard - Jetty Tuhup 🌊",
                              xaxis_title="Date", yaxis_title="Water Level (m)",
                              xaxis=dict(tickangle=90),
                              height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Terjadi error saat membaca file: {e}")

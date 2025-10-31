# WaterLevel_API_Jetty.py
import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Prediksi Level Air Jetty Tuhup 🌊")

# -----------------------------
# Upload CSV
# -----------------------------
st.subheader("Upload CSV Level Air")
uploaded_file = st.file_uploader("Pilih file CSV (Kolom: Datetime, Level Air)", type="csv")

if uploaded_file is not None:
    # Baca CSV
    df = pd.read_csv(uploaded_file)
    if "Datetime" not in df.columns or "Level Air" not in df.columns:
        st.error("CSV harus memiliki kolom 'Datetime' dan 'Level Air'")
    else:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").set_index("Datetime")
        df.rename(columns={"Level Air": "Water_level"}, inplace=True)

        st.subheader("Preview CSV")
        st.dataframe(df.style.format("{:.2f}").set_properties(**{"text-align":"right"}))

        # -----------------------------
        # Input tanggal prediksi
        # -----------------------------
        today = datetime.today().date()
        st.subheader("Pilih Tanggal Prediksi (H+1 s/d H+14)")
        pred_date = st.date_input(
            "Tanggal Prediksi",
            value=today + timedelta(days=1),
            min_value=today + timedelta(days=1),
            max_value=today + timedelta(days=14)
        )

        # -----------------------------
        # Tentukan lag Water_level
        # -----------------------------
        lag_days = 7
        wl_inputs = []
        for i in range(lag_days, 0, -1):
            day = pred_date - timedelta(days=i)
            default_val = df["Water_level"].get(day, 20.0)
            val = st.number_input(f"Level Air {day.strftime('%d-%m-%Y')}", value=float(default_val), format="%.2f")
            wl_inputs.append(val)

        if st.button("Forecast Level Air"):
            st.subheader("Forecasting Progress")
            progress_bar = st.progress(0)

            # -----------------------------
            # Ambil historis H0..H-(lag_days-1)
            # -----------------------------
            start_hist = (today - timedelta(days=lag_days-1)).isoformat()
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
            df_hist.set_index("time", inplace=True)

            # Forecast H+1..H+n
            n_days = (pred_date - today).days
            url_forecast = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude=-0.61&longitude=114.8&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
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
            df_forecast = df_forecast.set_index("time").iloc[1:n_days+1]

            df_all = pd.concat([df_hist, df_forecast]).drop_duplicates().sort_index()
            df_all["Water_level"] = None

            # Isi lag manual
            for i, d in enumerate([pred_date - timedelta(days=i) for i in range(lag_days,0,-1)]):
                if d in df_all.index:
                    df_all.loc[d, "Water_level"] = wl_inputs[i]

            water_level_lags = wl_inputs[:]
            results = {}
            forecast_dates = []

            for step in range(1, n_days+1):
                pred_day = today + timedelta(days=step)
                inp = {}

                # Fitur cuaca
                for i in range(1, 8):
                    date_i = pred_day - timedelta(days=i)
                    inp[f"Precipitation_lag{i}d"] = [df_all["precipitation_sum"].get(date_i, 0.0)]
                    inp[f"Relative_humidity_lag{i}d"] = [df_all["relative_humidity"].get(date_i, 0.0)]
                for i in range(1, 5):
                    date_i = pred_day - timedelta(days=i)
                    inp[f"Temperature_lag{i}d"] = [df_all["temperature_mean"].get(date_i, 0.0)]

                # Lag water level
                for i in range(1, 8):
                    inp[f"Water_level_lag{i}d"] = [water_level_lags[i-1]]

                input_data = pd.DataFrame(inp)[model.feature_names_in_].fillna(0.0)
                prediction = model.predict(input_data)[0]
                results[pred_day] = prediction
                water_level_lags = [prediction] + water_level_lags[:-1]

                if pred_day in df_all.index:
                    df_all.loc[pred_day, "Water_level"] = round(prediction,2)
                    forecast_dates.append(pred_day)

                progress_bar.progress(int((step/n_days)*100))

            st.success("Forecast selesai ✅")

            # Preview dengan highlight forecast
            def highlight_forecast(row):
                return ['background-color: #bce4f6' if row.name in forecast_dates else '' for _ in row]

            st.subheader("Preview Level Air (History + Forecast)")
            numeric_cols = ["Water_level","precipitation_sum","temperature_mean"]
            st.dataframe(df_all.style.apply(highlight_forecast, axis=1).format("{:.2f}", subset=numeric_cols).set_properties(**{"text-align":"right"}, subset=numeric_cols))

            # Plot
            lower_limit, upper_limit = 19.5, 26.5
            df_plot = df_all.reset_index().rename(columns={"time":"Date"})
            df_hist_plot = df_plot[df_plot["Date"] <= today]
            df_pred_plot = df_plot[df_plot["Date"] > today]
            df_pred_safe = df_pred_plot[df_pred_plot["Water_level"].between(lower_limit,upper_limit)]
            df_pred_unsafe = df_pred_plot[(df_pred_plot["Water_level"]<lower_limit)|(df_pred_plot["Water_level"]>upper_limit)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_hist_plot["Date"], y=df_hist_plot["Water_level"], mode="lines+markers", name="Historical", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df_pred_safe["Date"], y=df_pred_safe["Water_level"], mode="lines+markers", name="Prediction (Loadable)", line=dict(color="green", dash="dash")))
            fig.add_trace(go.Scatter(x=df_pred_unsafe["Date"], y=df_pred_unsafe["Water_level"], mode="lines+markers", name="Prediction (Unloadable)", line=dict(color="red", dash="dash")))

            fig.add_hline(y=lower_limit, line=dict(color="red", dash="dash"), annotation_text="Lower Limit", annotation_position="bottom left")
            fig.add_hline(y=upper_limit, line=dict(color="red", dash="dash"), annotation_text="Upper Limit", annotation_position="top left")

            st.plotly_chart(fig, use_container_width=True)

            # Download
            csv = df_all.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil Prediksi CSV", csv, "forecast_level_air.csv", "text/csv")

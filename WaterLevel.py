import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta, date
from io import BytesIO
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import plotly.graph_objects as go

# -----------------------------
# Load trained XGB model
# -----------------------------
model = joblib.load("best_model.pkl")
st.title("🌊 Water Level Daily Forecast Dashboard")

# -----------------------------
# Current time (GMT+7), rounded up to next day
# -----------------------------
now_utc = datetime.utcnow()
gmt7_now = now_utc + timedelta(hours=7)
max_date = (gmt7_now + timedelta(days=1)).date()  # maksimal besok

# -----------------------------
# Select forecast start datetime
# -----------------------------
st.subheader("Select Start Date for 14-Day Forecast")
selected_date = st.date_input(
    "Forecast Start Date",
    value=max_date,
    max_value=max_date,
    min_value=date(2019,1,1)
)
start_datetime = datetime.combine(selected_date, datetime.min.time())
st.write(f"Forecast start date (GMT+7): {start_datetime.date()}")

# -----------------------------
# Instructions for upload
# -----------------------------
st.subheader("Upload Water Level CSV")
st.info(
    f"- CSV must contain columns: 'Datetime' and 'Level Air'.\n"
    f"- 'Datetime' format: YYYY-MM-DD HH:MM:SS\n"
    f"- Data must cover **7 days before start date**: "
    f"{start_datetime.date() - timedelta(days=7)} to {start_datetime.date() - timedelta(days=1)}\n"
    f"- Make sure there are no duplicate timestamps."
)

# -----------------------------
# Upload water level data
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
wl_daily = None
upload_success = False

if uploaded_file is not None:
    try:
        df_wl = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
        if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
            st.error("The file must contain columns 'Datetime' and 'Level Air'.")
        else:
            # -----------------------
            # 1️⃣ Siapkan data harian
            # -----------------------
            df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"])
            df_wl = df_wl.drop_duplicates(subset="Datetime").sort_values("Datetime").set_index("Datetime")
            df_wl["Water_level"] = df_wl["Level Air"].clip(lower=0)
            
            # -----------------------
            # 2️⃣ Resample per hari
            # -----------------------
            wl_daily = df_wl.resample('D').mean().reset_index()
            wl_daily['Water_level'] = wl_daily['Water_level'].round(2)

            # -----------------------
            # 3️⃣ Ambil hanya 7 hari terakhir sebelum start
            # -----------------------
            start_limit = start_datetime - pd.Timedelta(days=7)
            wl_daily = wl_daily[(wl_daily["Datetime"] >= start_limit) & (wl_daily["Datetime"] < start_datetime)]
            wl_daily = wl_daily.reset_index(drop=True)

            # -----------------------
            # 4️⃣ Validasi jumlah hari
            # -----------------------
            expected_dates = pd.date_range(start=start_limit.date(), end=start_datetime.date()-timedelta(days=1), freq='D')
            actual_dates = pd.to_datetime(wl_daily["Datetime"]).dt.date
            missing_dates = sorted(set(expected_dates.date) - set(actual_dates))
            if missing_dates:
                missing_str = ', '.join([d.strftime("%Y-%m-%d") for d in missing_dates])
                st.warning(f"The uploaded water level data is incomplete! Missing dates: {missing_str}")
            else:
                upload_success = True
                st.success("✅ File uploaded and validated successfully!")
                st.dataframe(wl_daily)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# -----------------------------
# Run Forecast Button
# -----------------------------
run_forecast = st.button("Run 14-Day Forecast")
if "forecast_done" not in st.session_state:
    st.session_state["forecast_done"] = False
    st.session_state["final_df"] = None
    st.session_state["forecast_running"] = False

if run_forecast:
    st.session_state["forecast_done"] = False
    st.session_state["final_df"] = None
    st.session_state["forecast_running"] = True
    st.rerun()

# -----------------------------
# Forecast Logic
# -----------------------------
if upload_success and st.session_state["forecast_running"]:
    progress_container = st.empty()
    total_forecast_days = 14
    total_steps = 2 + total_forecast_days  # 2 step: prepare + validation, 14 days forecast
    step_counter = 0
    progress_bar = st.progress(0)

    # 1️⃣ Prepare historical
    progress_container.markdown("Preparing historical water level data...")
    historical_df = wl_daily.copy()
    historical_df["Source"] = "Historical"
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # 2️⃣ Prepare forecast DataFrame
    progress_container.markdown("Preparing forecast DataFrame...")
    forecast_dates = [start_datetime + timedelta(days=i) for i in range(total_forecast_days)]
    forecast_df = pd.DataFrame({"Datetime": forecast_dates})
    forecast_df["Water_level"] = np.nan
    forecast_df["Source"] = "Forecast"
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # 3️⃣ Iterative daily forecast
    progress_container.markdown("Running daily forecast iteratively...")
    final_df = pd.concat([historical_df, forecast_df], ignore_index=True).sort_values("Datetime").reset_index(drop=True)
    model_features = model.get_booster().feature_names

    forecast_indices = final_df.index[final_df["Source"]=="Forecast"]
    for i, idx in enumerate(forecast_indices, start=1):
        progress_container.markdown(f"Predicting day {i}/{total_forecast_days}...")
        X_forecast = pd.DataFrame(columns=model_features, index=[0])

        for f in model_features:
            if "_Lag" in f:
                base, lag_str = f.rsplit("_Lag",1)
                lag = int(lag_str)
            else:
                base = f
                lag = 0

            if base in final_df.columns:
                if idx-lag >= 0:
                    X_forecast.at[0,f] = final_df.iloc[idx-lag].get(base, 0)
                else:
                    X_forecast.at[0,f] = final_df[final_df["Source"]=="Historical"].iloc[0].get(base,0)
            else:
                X_forecast.at[0,f] = 0

        X_forecast = X_forecast.astype(float)
        y_hat = model.predict(X_forecast)[0]
        if y_hat < 0: y_hat = 0
        final_df.at[idx,"Water_level"] = round(y_hat,2)

        step_counter += 1
        progress_bar.progress(step_counter / total_steps)

    st.session_state["final_df"] = final_df
    st.session_state["forecast_done"] = True
    st.session_state["forecast_running"] = False
    progress_container.markdown("✅ 14-Day Water Level Forecast Completed!")
    progress_bar.progress(1.0)

# -----------------------------
# Display Forecast & Plot
# -----------------------------
result_container = st.empty()
if st.session_state.get("forecast_done") and st.session_state.get("final_df") is not None:
    final_df = st.session_state["final_df"]
    with result_container.container():
        st.subheader("Historical + Forecast Water Level + Climate Data")
        
        # Pilih kolom yang ditampilkan: Water_level + fitur iklim (tanpa lag)
        display_cols = ["Datetime", "Water_level", "Rainfall", "Cloud_cover", "Soil_moisture", "Source"]
        df_display = final_df[display_cols].copy()
        
        def highlight_forecast(row):
            return ['background-color: #cfe9ff' if row['Source']=="Forecast" else '' for _ in row]

        numeric_cols = df_display.select_dtypes(include=[np.number]).columns.tolist()
        styled_df = df_display.style.apply(highlight_forecast, axis=1)\
                                     .format({col: "{:.2f}" for col in numeric_cols})
        st.dataframe(styled_df, use_container_width=True, height=500)

        # -----------------------------
        # Plot
        # -----------------------------
        st.subheader("Water Level Historical vs Forecast")
        fig = go.Figure()
        hist_df = final_df[final_df["Source"]=="Historical"]
        fore_df = final_df[final_df["Source"]=="Forecast"]

        if not fore_df.empty:
            rmse = 0.05
            last_val = hist_df["Water_level"].iloc[-1]
            forecast_x = pd.concat([pd.Series([hist_df["Datetime"].iloc[-1]]), fore_df["Datetime"]])
            forecast_y = pd.concat([pd.Series([last_val]), fore_df["Water_level"]])
            upper_y = forecast_y + rmse
            lower_y = forecast_y - rmse
            lower_y = lower_y.clip(lower=0)

            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_x, forecast_x[::-1]]),
                y=pd.concat([upper_y, lower_y[::-1]]),
                fill="toself",
                fillcolor="rgba(255,165,0,0.2)",
                line=dict(color="rgba(255,165,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="±RMSE"
            ))
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="orange"),
                marker=dict(size=6)
            ))

        fig.add_trace(go.Scatter(
            x=hist_df["Datetime"],
            y=hist_df["Water_level"],
            mode="lines+markers",
            name="Historical",
            line=dict(color="blue"),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="Water Level Historical vs Forecast",
            xaxis_title="Date",
            yaxis_title="Water Level (m)",
            template="plotly_white",
            annotations=[
                dict(
                    xref="paper", yref="paper",
                    x=0.98, y=0.95,
                    text=f"RMSE = {rmse:.2f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    borderpad=4
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Download forecast only
        # -----------------------------
        forecast_only = final_df[final_df["Source"]=="Forecast"][["Datetime","Water_level"]].copy()
        forecast_only["Water_level"] = forecast_only["Water_level"].round(2)
        forecast_only["Datetime"] = forecast_only["Datetime"].astype(str)

        # CSV
        csv_buffer = forecast_only.to_csv(index=False).encode('utf-8')

        # Excel
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            forecast_only.to_excel(writer, index=False, sheet_name="Forecast")
        excel_buffer.seek(0)

        # PDF
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        data = [forecast_only.columns.tolist()] + forecast_only.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#007acc")),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
            ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ]))
        elements = [Paragraph("Joloi Water Level Forecast (Forecast Only)", styles["Title"]), table]
        doc.build(elements)

        # Download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download CSV", csv_buffer, "water_level_forecast.csv", "text/csv", use_container_width=True)
        with col2:
            st.download_button("Download Excel", excel_buffer, "water_level_forecast.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with col3:
            st.download_button("Download PDF", pdf_buffer.getvalue(), "water_level_forecast.pdf", "application/pdf", use_container_width=True)

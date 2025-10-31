# WaterLevel_API_hybrid_upload.py
import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# -----------------------------
# Load trained model (tetap pakai best_model.pkl)
# -----------------------------
model = joblib.load("best_model.pkl")

st.title("Water Level Prediction Dashboard 🌊 (Hybrid + Upload)")

today = datetime.today().date()

st.subheader("Pilih Tanggal Referensi / Target (untuk mode historis pilih tanggal lampau)")
pred_date = st.date_input(
    "Tanggal Referensi (digunakan sebagai titik awal jika memilih mode historis)",
    value=today,
    max_value=today + timedelta(days=14)
)

st.markdown("""
**Catatan:**  
- Upload file CSV berisi data *raw* water level (boleh berfrekuensi jam/menit).  
- File akan di-*aggregate* otomatis menjadi rata-rata harian (`Datetime` & `Water_level`).  
- Jika tidak upload, aplikasi akan meminta file sebelum menjalankan prediksi.
""")

# -----------------------------
# Upload CSV water level historis (harian akan dihitung secara otomatis)
# -----------------------------
st.subheader("Upload CSV Water Level Historis (wajib untuk validasi historis)")
uploaded_file = st.file_uploader("Upload file CSV (kolom: Datetime, Water_level)", type=["csv"])

wl_daily = None
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, engine="python", skip_blank_lines=True)
        if "Datetime" not in df_raw.columns or "Water_level" not in df_raw.columns:
            st.error("File CSV harus berisi kolom 'Datetime' dan 'Water_level'. Periksa nama kolom dan coba lagi.")
            st.stop()
        # parse datetime, aggregate ke harian (mean)
        df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"])
        df_raw["Date"] = df_raw["Datetime"].dt.date
        df_daily = df_raw.groupby("Date", as_index=False)["Water_level"].mean().round(3)
        df_daily.rename(columns={"Date": "Datetime"}, inplace=True)
        df_daily["Datetime"] = pd.to_datetime(df_daily["Datetime"]).dt.date  # keep date type
        df_daily.set_index("Datetime", inplace=True)
        wl_daily = df_daily.copy()
        st.success("✅ File water level berhasil diproses (rata-rata harian).")
        st.dataframe(wl_daily.tail(14).style.format("{:.3f}"))
    except Exception as e:
        st.error(f"Gagal memproses file: {e}")
        st.stop()
else:
    st.warning("Silakan upload CSV water level historis untuk melanjutkan.")
    st.stop()

# -----------------------------
# fitur model (tetap)
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

# -----------------------------
# helper safe get
# -----------------------------
def safe_get(df, date_key, col):
    try:
        return float(df.loc[date_key, col])
    except Exception:
        return 0.0

# -----------------------------
# Tentukan start_point & periode yg diperlukan
# Rule:
# - Jika user memilih pred_date <= today => mode HISTORIS,
#   start_point = pred_date
# - Jika user memilih pred_date > today => mode FORECAST,
#   start_point = today + 1 (mulai prediksi esok hari)
# Selalu lakukan prediksi untuk 14 hari dimulai dari start_point
# -----------------------------
if pred_date <= today:
    start_point = pred_date  # mode historis validation: mulai prediksi dari pred_date
    mode = "historical"
else:
    start_point = today + timedelta(days=1)  # mode forecast: mulai besok
    mode = "forecast"

forecast_horizon = 14  # prediksi 14 hari
# Untuk mendapatkan lag cuaca kita perlu data dari start_point-7 ... start_point+forecast_horizon-1
climate_start = start_point - timedelta(days=7)
climate_end = start_point + timedelta(days=forecast_horizon - 1)

st.info(f"Mode: **{mode}** — melakukan prediksi {forecast_horizon} hari mulai {start_point.isoformat()} sampai {(start_point + timedelta(days=forecast_horizon-1)).isoformat()}")

# -----------------------------
# Ambil data iklim: historical (archive) + forecast (jika perlu)
# Kita akan mengumpulkan daily climate untuk rentang climate_start..climate_end
# -----------------------------
def fetch_climate_range(start_date, end_date):
    # fetch historical archive for portion up to today (if any)
    dfs = []
    # historical portion
    hist_start = start_date
    hist_end = min(end_date, today)
    if hist_start <= hist_end:
        url_hist = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=-0.61&longitude=114.8&start_date={hist_start.isoformat()}&end_date={hist_end.isoformat()}"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        try:
            r = requests.get(url_hist, timeout=30).json()
            df_hist = pd.DataFrame({
                "time": r["daily"]["time"],
                "precipitation_sum": r["daily"]["precipitation_sum"],
                "temperature_mean": r["daily"]["temperature_2m_mean"],
                "relative_humidity": r["daily"]["relative_humidity_2m_mean"]
            })
            df_hist["time"] = pd.to_datetime(df_hist["time"]).dt.date
            df_hist.set_index("time", inplace=True)
            dfs.append(df_hist)
        except Exception as e:
            st.warning(f"Gagal mengambil historical climate: {e}")
    # forecast portion (if need beyond today)
    if end_date > today:
        # fetch forecast (we'll slice later)
        url_fore = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude=-0.61&longitude=114.8"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore&forecast_days=16"
        )
        try:
            r2 = requests.get(url_fore, timeout=30).json()
            df_fore = pd.DataFrame({
                "time": r2["daily"]["time"],
                "precipitation_sum": r2["daily"]["precipitation_sum"],
                "temperature_mean": r2["daily"]["temperature_2m_mean"],
                "relative_humidity": r2["daily"]["relative_humidity_2m_mean"]
            })
            df_fore["time"] = pd.to_datetime(df_fore["time"]).dt.date
            df_fore.set_index("time", inplace=True)
            # select only dates > today and up to end_date
            df_fore = df_fore.loc[(df_fore.index > today) & (df_fore.index <= end_date)]
            dfs.append(df_fore)
        except Exception as e:
            st.warning(f"Gagal mengambil forecast climate: {e}")

    if dfs:
        df_all = pd.concat(dfs).drop_duplicates().sort_index()
    else:
        df_all = pd.DataFrame(columns=["precipitation_sum", "temperature_mean", "relative_humidity"])
    return df_all

# fetch climate for needed range
with st.spinner("Mengambil data iklim (historis & forecast bila perlu)..."):
    df_climate = fetch_climate_range(climate_start, climate_end)

if df_climate.empty:
    st.warning("Data iklim kosong untuk rentang yang dibutuhkan. Pastikan koneksi API bekerja.")
    st.stop()

# -----------------------------
# Integrasi water level historis (dari upload) ke df_climate
# - wl_daily index = date
# - df_climate index = date
# We'll create df_all covering climate_start..climate_end and attach water_level where available
# -----------------------------
# create full date index
all_dates = pd.date_range(climate_start, climate_end, freq="D").date
df_all = pd.DataFrame(index=all_dates)
# attach climate columns
for col in df_climate.columns:
    df_all[col] = df_climate.reindex(all_dates)[col].values
# attach water_level from wl_daily
df_all["water_level"] = None
for d, row in wl_daily.iterrows():
    # wl_daily index is date
    if d in df_all.index:
        df_all.at[d, "water_level"] = float(row["Water_level"])

# If wl_daily has dates beyond climate_end or before climate_start, they are ignored here.
# -----------------------------
# Prepare initial water level lags (7 days before start_point)
# We need water_level for start_point-1 .. start_point-7
# We'll extract from wl_daily; if a day missing, try to fill by interpolation using wl_daily (reindex & interpolate)
# -----------------------------
lag_dates = [(start_point - timedelta(days=i)).isoformat() for i in range(1, 8)]
# build series of daily WL from wl_daily for range (start_point - 14 .. start_point -1) to allow interpolation
wl_index_start = start_point - timedelta(days=14)
wl_index_end = start_point - timedelta(days=1)
wl_index = pd.date_range(wl_index_start, wl_index_end, freq="D").date
wl_series = pd.Series(index=wl_index, dtype=float)
for d in wl_index:
    try:
        wl_series.loc[d] = float(wl_daily.loc[d, "Water_level"])
    except Exception:
        wl_series.loc[d] = None
# interpolate and fill
wl_series = wl_series.interpolate().bfill().ffill().fillna(0.0)
# take last 7 values (most recent first order: WL_lag1d = pred_day-1)
water_level_lags = [float(wl_series.loc[start_point - timedelta(days=i)]) for i in range(1, 8)]

# -----------------------------
# Iterative autoregressive forecast for 14 days starting at start_point
# For each pred_day in start_point .. start_point+13:
#   - build features using climate lags and WL lags
#   - predict using model
#   - append predicted WL into df_all and update water_level_lags
# -----------------------------
results = {}
forecast_dates = []

for step in range(0, forecast_horizon):
    pred_day = start_point + timedelta(days=step)  # date
    # prepare input dictionary
    inp = {}
    # climate lags: Precipitation_lag1d .. Precipitation_lag7d (pred_day - i)
    for i in range(1, 8):
        date_i = pred_day - timedelta(days=i)
        inp[f"Precipitation_lag{i}d"] = [safe_get(df_all, date_i, "precipitation_sum")]
    # temperature lags 1..4
    for i in range(1, 5):
        date_i = pred_day - timedelta(days=i)
        inp[f"Temperature_lag{i}d"] = [safe_get(df_all, date_i, "temperature_mean")]
    # relative humidity lags 1..7
    for i in range(1, 8):
        date_i = pred_day - timedelta(days=i)
        inp[f"Relative_humidity_lag{i}d"] = [safe_get(df_all, date_i, "relative_humidity")]
    # water level lags (WL_lag1d .. WL_lag7d) from water_level_lags (which is ordered [H-1, H-2, ...])
    for i in range(1, 8):
        inp[f"Water_level_lag{i}d"] = [water_level_lags[i-1]]

    input_data = pd.DataFrame(inp)
    # ensure column order matches features list; if some feature missing, fill 0
    input_data = input_data.reindex(columns=features, fill_value=0.0)

    # predict
    try:
        prediction = model.predict(input_data)[0]
        # enforce non-negative
        if prediction is None:
            prediction = 0.0
        if prediction < 0:
            prediction = 0.0
    except Exception as e:
        st.warning(f"Model predict error on {pred_day}: {e}")
        prediction = 0.0

    # store result
    results[pred_day] = round(float(prediction), 3)
    forecast_dates.append(pred_day)

    # update df_all
    if pred_day in df_all.index:
        df_all.at[pred_day, "water_level"] = round(float(prediction), 3)
    else:
        df_all.loc[pred_day, "water_level"] = round(float(prediction), 3)

    # update water_level_lags for next iteration
    water_level_lags = [prediction] + water_level_lags[:-1]

# -----------------------------
# Prepare preview dataframe similar to original (with climate columns and water_level first)
# -----------------------------
df_preview = df_all.copy()
# ensure columns exist
for c in ["precipitation_sum", "temperature_mean", "relative_humidity", "water_level"]:
    if c not in df_preview.columns:
        df_preview[c] = None
# move water_level to first column for preview
cols = df_preview.columns.tolist()
if "water_level" in cols:
    cols.insert(0, cols.pop(cols.index("water_level")))
df_preview = df_preview[cols]
# convert index to column "time" to match original plotting pipeline
df_preview = df_preview.reset_index().rename(columns={"index": "time"})
# format numeric columns
numeric_cols = ["precipitation_sum", "temperature_mean", "water_level"]
st.subheader("Preview Data (Climate + Water Level — termasuk hasil prediksi 14 hari)")
st.dataframe(df_preview.style.format("{:.3f}", subset=numeric_cols).set_properties(**{"text-align": "right"}, subset=numeric_cols))

# -----------------------------
# Show summary predicted values
# -----------------------------
st.subheader("Ringkasan Prediksi 14 Hari")
for d, v in results.items():
    st.write(f"{d.strftime('%Y-%m-%d')}: {v:.3f} m")

# -----------------------------
# PLOT SECTION (mirip dengan plot di file lama, menampilkan 14 hari)
# -----------------------------
# Build df_plot same as original: columns time, water_level
df_plot = df_preview[["time", "water_level"]].copy()
df_plot.rename(columns={"time": "Date"}, inplace=True)
df_plot["Date"] = pd.to_datetime(df_plot["Date"]).dt.date
# ensure numeric
df_plot["water_level"] = pd.to_numeric(df_plot["water_level"], errors="coerce")

# loadable limits (sesuaikan jika perlu)
lower_limit = 19.5
upper_limit = 26.5

# split historical vs prediction relative to today
df_hist = df_plot[df_plot["Date"] <= today]
df_pred = df_plot[df_plot["Date"] > today]

df_pred_safe = df_pred[df_pred["water_level"].between(lower_limit, upper_limit)]
df_pred_unsafe = df_pred[(df_pred["water_level"] < lower_limit) | (df_pred["water_level"] > upper_limit)]

fig = go.Figure()

# Historical line
fig.add_trace(go.Scatter(
    x=df_hist["Date"],
    y=df_hist["water_level"],
    mode="lines",
    line=dict(color="blue", width=2),
    showlegend=False,
    hoverinfo="skip"
))
fig.add_trace(go.Scatter(
    x=df_hist["Date"],
    y=df_hist["water_level"],
    mode="markers",
    marker=dict(color="blue", size=8),
    showlegend=True,
    name="Historical"
))

# dashed continuity: connect last historical point (continuity_start) with predictions
continuity_start = start_point - timedelta(days=1) if start_point - timedelta(days=1) in df_plot["Date"].values else None
last_hist_val = None
if continuity_start is not None:
    last_hist_row = df_plot[df_plot["Date"] == continuity_start]
    if not last_hist_row.empty and pd.notna(last_hist_row["water_level"].iloc[0]):
        last_hist_val = float(last_hist_row["water_level"].iloc[0])

# pred numeric
df_pred_nonnull = df_pred.dropna(subset=["water_level"]).copy()
pred_dates_numeric = df_pred_nonnull["Date"].tolist()
pred_values_numeric = df_pred_nonnull["water_level"].tolist()

dashed_x = []
dashed_y = []
if last_hist_val is not None and len(pred_dates_numeric) >= 1:
    dashed_x = [continuity_start] + pred_dates_numeric
    dashed_y = [last_hist_val] + pred_values_numeric
elif len(pred_dates_numeric) >= 2:
    dashed_x = pred_dates_numeric
    dashed_y = pred_values_numeric

if len(dashed_x) >= 2:
    fig.add_trace(go.Scatter(
        x=dashed_x,
        y=dashed_y,
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))

# prediction safe & unsafe
fig.add_trace(go.Scatter(
    x=df_pred_safe["Date"],
    y=df_pred_safe["water_level"],
    mode="markers",
    marker=dict(color="green", size=8),
    name="Prediction (Loadable)"
))
fig.add_trace(go.Scatter(
    x=df_pred_unsafe["Date"],
    y=df_pred_unsafe["water_level"],
    mode="markers",
    marker=dict(color="red", size=8),
    name="Prediction (Unloadable)"
))

# Today marker
today_point = df_plot[df_plot["Date"] == today]
if not today_point.empty and pd.notna(today_point["water_level"].iloc[0]):
    fig.add_trace(go.Scatter(
        x=today_point["Date"],
        y=today_point["water_level"],
        mode="markers",
        marker=dict(color="blue", size=8, symbol="circle"),
        name="Today",
        showlegend=False
    ))

# Limits lines
fig.add_hline(y=lower_limit, line=dict(color="red", width=2, dash="dash"),
              annotation_text="Lower Limit", annotation_position="bottom left")
fig.add_hline(y=upper_limit, line=dict(color="red", width=2, dash="dash"),
              annotation_text="Upper Limit", annotation_position="top left")

# RMSE band (gunakan last_hist_val as anchor jika tersedia)
rmse = 0.87
band_added = False
if last_hist_val is not None and len(pred_dates_numeric) >= 1:
    band_x = [continuity_start] + pred_dates_numeric
    upper_band = [last_hist_val + rmse] + [v + rmse for v in pred_values_numeric]
    lower_band = [last_hist_val - rmse] + [v - rmse for v in pred_values_numeric]
    band_added = True
elif len(pred_values_numeric) >= 2:
    band_x = pred_dates_numeric
    upper_band = [v + rmse for v in pred_values_numeric]
    lower_band = [v - rmse for v in pred_values_numeric]
    band_added = True

if band_added:
    fig.add_trace(go.Scatter(
        x=band_x,
        y=lower_band,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=band_x,
        y=upper_band,
        mode="lines",
        line=dict(color="rgba(0,0,0,0)", dash="dash"),
        fill="tonexty",
        fillcolor="rgba(0, 0, 255, 0.12)",
        name=f"Prediction error ±{rmse:.2f} m",
        showlegend=True
    ))

# Format ticks
all_dates = df_plot["Date"]
tick_text = [d.strftime("%d/%m/%y") for d in all_dates]

fig.update_layout(
    title="Water Level Dashboard 🌊 (14-day forecast window)",
    xaxis_title="Date",
    yaxis_title="Water Level (m)",
    xaxis=dict(
        tickangle=90,
        tickmode="array",
        tickvals=all_dates,
        ticktext=tick_text
    ),
    yaxis=dict(autorange=True),
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Optional: provide CSV download for the 14-day forecast portion
# -----------------------------
forecast_out_df = df_preview[df_preview["time"].dt.date.isin(forecast_dates)][["time", "water_level"]].copy()
forecast_out_df.rename(columns={"time": "Date", "water_level": "Predicted_Water_level"}, inplace=True)
forecast_out_df["Date"] = forecast_out_df["Date"].dt.strftime("%Y-%m-%d")
csv_bytes = forecast_out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, "waterlevel_14day_forecast.csv", "text/csv", use_container_width=True)

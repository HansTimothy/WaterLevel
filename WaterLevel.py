# -----------------------------
# Fetch data & predict
# -----------------------------
if st.button("Fetch Data & Predict"):

    if pred_date <= today + timedelta(days=1):
        # ---------------- HISTORICAL CASE (H-7 .. H+1) ----------------
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=-0.61&longitude=114.8&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame({
            "time": data["daily"]["time"],
            "precipitation_sum": data["daily"]["precipitation_sum"],
            "temperature_mean": data["daily"]["temperature_2m_mean"],
            "relative_humidity": data["daily"]["relative_humidity_2m_mean"]
        })
        df["time"] = pd.to_datetime(df["time"]).dt.date
        df.set_index("time", inplace=True)

        st.subheader("Preview API Data")
        st.dataframe(df)

        # Buat input feature
        input_data = pd.DataFrame({
            **{f"Precipitation_lag{i}d": [df["precipitation_sum"].iloc[-i]] for i in range(1, 8)},
            **{f"Temperature_lag{i}d": [df["temperature_mean"].iloc[-i]] for i in range(1, 5)},
            **{f"Relative_humidity_lag{i}d": [df["relative_humidity"].iloc[-i]] for i in range(1, 8)},
            **{f"Water_level_lag{i}d": [wl_inputs[i-1]] for i in range(1, 8)}
        })

        input_data = input_data[features].fillna(0.0)

        # Prediksi
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Water Level on {pred_date}: {prediction:.2f} m")

    else:
        # ---------------- FORECAST CASE (H+2 .. H+7) ----------------
        # Ambil data historis (H-6 .. today)
        url_hist = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=-0.61&longitude=114.8&start_date={today - timedelta(days=6)}&end_date={today}"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        response_hist = requests.get(url_hist)
        data_hist = response_hist.json()
        df_hist = pd.DataFrame({
            "time": data_hist["daily"]["time"],
            "precipitation_sum": data_hist["daily"]["precipitation_sum"],
            "temperature_mean": data_hist["daily"]["temperature_2m_mean"],
            "relative_humidity": data_hist["daily"]["relative_humidity_2m_mean"]
        })
        df_hist["time"] = pd.to_datetime(df_hist["time"]).dt.date

        # Ambil data forecast (today+1 .. today+7)
        url_fore = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude=-0.61&longitude=114.8"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        response_fore = requests.get(url_fore)
        data_fore = response_fore.json()
        df_fore = pd.DataFrame({
            "time": data_fore["daily"]["time"],
            "precipitation_sum": data_fore["daily"]["precipitation_sum"],
            "temperature_mean": data_fore["daily"]["temperature_2m_mean"],
            "relative_humidity": data_fore["daily"]["relative_humidity_2m_mean"]
        })
        df_fore["time"] = pd.to_datetime(df_fore["time"]).dt.date

        # Gabung historis + forecast
        df_all = pd.concat([df_hist, df_fore]).set_index("time")

        # Loop prediksi step by step
        wl_series = {wl_dates[i]: wl_inputs[i] for i in range(7)}  # manual input
        predictions = {}

        current_date = today + timedelta(days=1)
        while current_date <= pred_date:
            idx = df_all.index.get_loc(current_date)

            input_data = pd.DataFrame({
                **{f"Precipitation_lag{i}d": [df_all["precipitation_sum"].iloc[idx - i]] for i in range(1, 8)},
                **{f"Temperature_lag{i}d": [df_all["temperature_mean"].iloc[idx - i]] for i in range(1, 5)},
                **{f"Relative_humidity_lag{i}d": [df_all["relative_humidity"].iloc[idx - i]] for i in range(1, 8)},
                **{f"Water_level_lag{i}d": [wl_series[current_date - timedelta(days=i)]] for i in range(1, 8)}
            })
            input_data = input_data[features].fillna(0.0)

            # Prediksi untuk current_date
            pred_val = model.predict(input_data)[0]
            wl_series[current_date] = pred_val
            predictions[current_date] = pred_val

            current_date += timedelta(days=1)

        # Tampilkan preview hanya sampai tanggal dipilih
        st.subheader("Preview API Data")
        st.dataframe(df_all.loc[:pred_date])

        # Hanya tampilkan hasil prediksi untuk tanggal target
        final_pred = predictions[pred_date]
        st.success(f"Predicted Water Level on {pred_date}: {final_pred:.2f} m")

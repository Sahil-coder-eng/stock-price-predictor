import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import time

# Page config
st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")
st.markdown("<h1 style='text-align: center;'>📈 Stock Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("Use AI (LSTM) to predict stock prices using historical data from Yahoo Finance.")

# Sidebar input
st.sidebar.title("⚙️ App Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TATAMOTORS.NS)", "TATAMOTORS.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
user_end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-30"))
epochs = st.sidebar.slider("Training Epochs", 1, 20, 5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64])

# Limit end date to today
today = datetime.date.today()
end_date = min(user_end_date, today)

# Predict button
if st.button("🔍 Predict Prices"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.warning("⚠️ No data found! Check your ticker or date range.")
    else:
        st.subheader(f"📊 Raw Data for {ticker.upper()} up to {end_date}")
        st.write(df.tail())

        # Prepare data
        df = df[['Close']]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i - 60:i])
            y.append(scaled_data[i])
        X, y = np.array(X), np.array(y)

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        with st.spinner("🧠 Training LSTM model..."):
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            time.sleep(1)

        # Predict and inverse transform
        predicted = model.predict(X)
        predicted_prices = scaler.inverse_transform(predicted)
        actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

        # Plot actual vs predicted
        st.subheader("📈 Predicted vs Actual Closing Prices")
        fig, ax = plt.subplots()
        ax.plot(actual_prices, label='Actual Price', linewidth=2)
        ax.plot(predicted_prices, label='Predicted Price', linestyle='--')
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        st.success("✅ Model trained and predictions done!")
        st.balloons()

        # Forecast next 7 days
        st.subheader("📅 Next 7-Day Forecast")
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        forecasted = []

        for _ in range(7):
            pred = model.predict(last_60_days)[0][0]
            forecasted.append(pred)
            new_input = np.append(last_60_days[0, 1:], [[pred]], axis=0)
            last_60_days = new_input.reshape(1, 60, 1)

        forecasted_prices = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))
        future_dates = pd.date_range(end_date + pd.Timedelta(days=1), periods=7)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close Price": forecasted_prices.flatten()
        }).set_index("Date")

        st.dataframe(forecast_df)

        fig2, ax2 = plt.subplots()
        ax2.plot(forecast_df.index, forecast_df["Predicted Close Price"], marker='o', color='green')
        ax2.set_title("Next 7 Days Stock Price Forecast")
        ax2.set_ylabel("Price")
        ax2.set_xlabel("Date")
        st.pyplot(fig2)
        st.markdown("### 📊 Model Summary")
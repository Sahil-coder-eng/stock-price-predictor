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

# Streamlit Page Configuration
st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")
st.markdown("<h1 style='text-align: center;'>📈 Stock Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("Use an LSTM model trained on historical stock data from Yahoo Finance to predict and forecast stock prices.")

# Sidebar Inputs
st.sidebar.title("⚙️ App Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TATAMOTORS.NS)", "TATAMOTORS.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
user_end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-07-30"))
epochs = st.sidebar.slider("Training Epochs", 1, 20, 5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64])

# Ensure end date does not exceed today
today = datetime.date.today()
end_date = min(user_end_date, today)

# Button to trigger prediction
if st.button("🔍 Predict Prices"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty or 'Close' not in df.columns or df['Close'].isnull().all():
        st.warning("⚠️ Invalid data fetched. Please check ticker symbol or date range.")
    else:
        st.subheader(f"📊 Closing Prices for {ticker.upper()}")
        st.line_chart(df['Close'])

        # Only use the 'Close' column
        close_data = df[['Close']].copy()

        # Normalize using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        # Create sequences
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(scaled_data[i])
        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            st.error("📉 Not enough data to train. Try increasing the date range.")
        else:
            # Reshape for LSTM input
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train model
            with st.spinner("🧠 Training LSTM model..."):
                model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
                time.sleep(1)

            # Predict
            predicted = model.predict(X)
            predicted_prices = scaler.inverse_transform(predicted)
            actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

            # Plot actual vs predicted
            st.subheader("📈 Actual vs Predicted Closing Prices")
            fig, ax = plt.subplots()
            ax.plot(actual_prices, label='Actual', linewidth=2)
            ax.plot(predicted_prices, label='Predicted', linestyle='--')
            ax.set_xlabel("Days")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # Forecast next 7 days
            st.subheader("📅 Next 7-Day Forecast")
            forecast_input = scaled_data[-60:].reshape(1, 60, 1)
            forecast = []

            for _ in range(7):
                next_pred = model.predict(forecast_input)[0][0]
                forecast.append(next_pred)
                forecast_input = np.append(forecast_input[:, 1:, :], [[[next_pred]]], axis=1)

            forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            future_dates = pd.date_range(end=close_data.index[-1] + pd.Timedelta(days=1), periods=7)

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Close Price": forecast_prices.flatten()
            }).set_index("Date")

            st.dataframe(forecast_df)

            # Forecast plot
            fig2, ax2 = plt.subplots()
            ax2.plot(forecast_df.index, forecast_df["Predicted Close Price"], marker='o', color='green')
            ax2.set_title("Next 7 Days Forecast")
            ax2.set_ylabel("Price")
            ax2.set_xlabel("Date")
            st.pyplot(fig2)

            st.success("✅ Prediction & Forecast completed successfully!")
            st.balloons()

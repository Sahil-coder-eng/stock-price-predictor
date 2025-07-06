import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")
st.markdown("<h1 style='text-align: center;'>📈 Stock Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("This app uses Linear Regression on historical stock data from Yahoo Finance to predict stock closing prices and forecast the next 7 days.")

# Sidebar inputs
st.sidebar.title("⚙️ App Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TATAMOTORS.NS)", "TATAMOTORS.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
user_end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-07-30"))

# Ensure end date is not in the future
today = datetime.date.today()
end_date = min(user_end_date, today)

if user_end_date > today:
    st.info(f"📅 You've selected a future end date ({user_end_date}), but data is only available up to today ({today}).")

# Predict button
if st.button("🔍 Predict Prices"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty or df['Close'].isnull().all():
        st.warning("⚠️ No data found for the selected date range or stock ticker.")
    else:
        st.subheader(f"📊 Closing Price for {ticker.upper()}")
        st.line_chart(df['Close'])

        # Prepare data for prediction
        df = df[['Close']].reset_index()
        df['Days'] = np.arange(len(df)).reshape(-1, 1)

        X = df[['Days']]
        y = df['Close']

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        predicted = model.predict(X)

        # Plot actual vs predicted
        st.subheader("📈 Predicted vs Actual Closing Prices")
        fig, ax = plt.subplots()
        ax.plot(df['Date'], y, label='Actual Price', linewidth=2)
        ax.plot(df['Date'], predicted, label='Predicted Price', linestyle='--')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Forecast next 7 days
        st.subheader("📅 Next 7-Day Forecast")
        last_day = df['Days'].iloc[-1]
        future_days = np.arange(last_day + 1, last_day + 8).reshape(-1, 1)
        future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)

        forecast = model.predict(future_days)
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close Price": forecast
        }).set_index("Date")

        st.dataframe(forecast_df)

        # Plot forecast
        fig2, ax2 = plt.subplots()
        ax2.plot(forecast_df.index, forecast_df["Predicted Close Price"], marker='o', color='green')
        ax2.set_title("Next 7 Days Stock Price Forecast")
        ax2.set_ylabel("Price")
        ax2.set_xlabel("Date")
        st.pyplot(fig2)

        st.success("✅ Forecast completed successfully!")

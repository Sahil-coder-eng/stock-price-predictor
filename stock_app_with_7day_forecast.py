import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Streamlit config
st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")
st.markdown("<h1 style='text-align: center;'>📈 Stock Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("This app uses Linear Regression on historical stock data from Yahoo Finance to predict closing prices and forecast the next 7 days.")

# Sidebar inputs
st.sidebar.title("⚙️ App Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TATAMOTORS.NS)", "TATAMOTORS.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
user_end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-07-30"))

# Ensure end date isn't in the future
today = datetime.date.today()
end_date = min(user_end_date, today)

# Predict button
if st.button("🔍 Predict Prices"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.warning("⚠️ No data found! Check your ticker or date range.")
    elif 'Close' not in df.columns:
        st.warning("⚠️ 'Close' column is missing in the data.")
    elif df['Close'].isnull().all():
        st.warning("⚠️ All values in 'Close' column are NaN.")
    else:
        st.subheader(f"📊 Closing Price for {ticker.upper()}")
        st.line_chart(df['Close'])

        df = df[['Close']].reset_index()
        df['Days'] = np.arange(len(df))

        # Model training
        X = df[['Days']]
        y = df['Close']
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

        fig2, ax2 = plt.subplots()
        ax2.plot(forecast_df.index, forecast_df["Predicted Close Price"], marker='o', color='green')
        ax2.set_title("Next 7 Days Stock Price Forecast")
        ax2.set_ylabel("Price")
        ax2.set_xlabel("Date")
        st.pyplot(fig2)

        st.success("✅ Forecast completed successfully!")
        st.balloons()
        st.markdown("### 📊 Model Summary")

import streamlit as st
import pandas as pd
import numpy as np
from data_processing import fetch_stock_data, add_technical_indicators, preprocess_data
from model import prepare_lstm_data, train_lstm_model, evaluate_regression_model, predict_future_lstm
from utils import plot_predictions

def main():
    st.title("Stock Market Trend Prediction")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="AAPL")
    period = st.selectbox("Select Data Period:", options=['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)
    interval = st.selectbox("Select Data Interval:", options=['1d', '1wk', '1mo'], index=0)
    future_days = st.number_input("Days to Predict into the Future:", min_value=1, max_value=30, value=5)

    if st.button("Fetch Data and Train Model"):
        with st.spinner("Fetching data..."):
            data = fetch_stock_data(ticker, period=period, interval=interval)
        if data.empty:
            st.error("No data found for the ticker.")
            return

        with st.spinner("Processing data..."):
            data_with_indicators = add_technical_indicators(data)
            processed_data = preprocess_data(data_with_indicators)

        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
                        'RSI_14', 'MACD', 'MACD_Signal']
        target_col = 'Close'

        with st.spinner("Preparing data for LSTM..."):
            X, y = prepare_lstm_data(processed_data, feature_cols, target_col, time_steps=60)

        # Split data into train and test sets
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        with st.spinner("Training LSTM model..."):
            model, history = train_lstm_model(X_train, y_train, epochs=10, batch_size=32)

        with st.spinner("Evaluating model..."):
            rmse, y_pred = evaluate_regression_model(model, X_test, y_test)
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        st.subheader("Prediction vs Actual")
        plot_predictions(processed_data.index[-len(y_test):], y_test, y_pred)

        # Future predictions
        last_sequence = X[-1]
        future_preds = predict_future_lstm(model, last_sequence, future_steps=future_days)

        # Generate future dates
        last_date = processed_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')  # Business days

        st.subheader(f"Future {future_days}-Day Predictions")
        st.write(pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds}))

        # Plot future predictions appended to actual data
        all_dates = future_dates
        all_actual = [np.nan] * future_days
        all_predicted = future_preds

        plot_predictions(all_dates, all_actual, all_predicted, title="Future Predicted Prices")

if __name__ == "__main__":
    main()

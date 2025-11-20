import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features  # Requires 'ta' library; pip install ta
from torch import nn


# For sentiment: Use X semantic search for real sentiment analysis via LLM
def get_sentiment_score():
    # Placeholder: In production, integrate with X API or semantic search to fetch recent BTC posts
    # Then use LLM to score average sentiment (-1 to 1)
    # For demo, simulate with random

    return random.uniform(-1, 1)  # Bearish to bullish


def prepare_data(data, window=30, test_size=0.2):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - window):
        X.append(scaled_data[i : i + window])
        y.append(scaled_data[i + window, 0])  # Predict close price
    X, y = np.array(X), np.array(y)
    split = int(len(X) * (1 - test_size))
    X_train, X_test = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
    y_train, y_test = torch.FloatTensor(y[:split]), torch.FloatTensor(y[split:])
    return X_train, y_train, X_test, y_test, scaler, scaled_data, split


class LSTMModel(nn.Module):
    def __init__(
        self, input_size=4, hidden_size=50, num_layers=1
    ):  # input_size for features
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def train_model(btc, window=30, epochs=50, lr=0.001):
    features_df = add_all_ta_features(
        btc,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )
    features_df = features_df[
        ["Close", "trend_sma_fast", "momentum_rsi", "volume_obv"]
    ].dropna()
    features = features_df.values
    if len(features) < window + 1:
        raise ValueError(f"❌ Not enough data (need at least {window + 1} rows).")
    X_train, y_train, X_test, y_test, scaler, scaled_data, split = prepare_data(
        features, window
    )
    model = LSTMModel(input_size=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Evaluate on original scale
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test).numpy().flatten()
        y_test_np = y_test.numpy()
        # Inverse scale using Close's params (feature 0)
        close_scale = scaler.scale_[0]
        close_min = scaler.min_[0]
        inverted_y_test = (y_test_np / close_scale) + close_min
        inverted_y_pred = (y_pred_test / close_scale) + close_min
        rmse = np.sqrt(mean_squared_error(inverted_y_test, inverted_y_pred))
        mape = (
            mean_absolute_percentage_error(inverted_y_test, inverted_y_pred) * 100
            if np.all(inverted_y_test != 0)
            else np.nan
        )

    return (
        model,
        scaler,
        X_train,
        y_train,
        X_test,
        y_test,
        inverted_y_pred,
        rmse,
        mape,
        features_df,
        inverted_y_test,
        close_scale,
        close_min,
        split,
        scaled_data,
    )


def backtest(features_df, inverted_y_pred, y_test_len):
    close_t = features_df["Close"].iloc[-y_test_len - 1 : -1].values
    signals = np.sign(inverted_y_pred - close_t)
    pct_returns = features_df["Close"].pct_change().iloc[-y_test_len:].values
    returns = signals * pct_returns
    returns = returns[~np.isnan(returns)]  # Remove any NaNs (shouldn't be any)
    cumulative = np.cumprod(1 + returns) - 1
    sharpe = (
        (np.mean(returns) / np.std(returns) * np.sqrt(252))
        if np.std(returns) > 0
        else 0
    )
    total_return = cumulative[-1] if len(cumulative) > 0 else 0
    return sharpe, total_return


def load_data(period="10y") -> pd.DataFrame:
    btc = yf.Ticker("BTC-USD").history(period=period)
    if btc.empty:
        raise ValueError("❌ Failed to download BTC-USD data.")
    return btc


def main():
    st.title("📈 Advanced Bitcoin Algo Trading Predictor with LLM & DL")
    st.markdown(
        "Leveraging deep learning (LSTM) for sequence modeling and LLM-driven sentiment from X for adjusted predictions. Includes backtesting for trading strategies."
    )

    # User inputs
    period = st.selectbox("Data Period", ["1y", "5y", "10y", "max"], index=2)
    window = st.slider("Sequence Window", 10, 60, 30)
    epochs = st.slider("Training Epochs", 20, 200, 50)

    # Load data
    try:
        btc = load_data(period)
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    st.subheader("Historical BTC Prices")
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Candlestick(
            x=btc.index,
            open=btc["Open"],
            high=btc["High"],
            low=btc["Low"],
            close=btc["Close"],
            name="BTC",
        )
    )
    fig_hist.update_layout(
        title="BTC-USD Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Train model
    try:
        (
            model,
            scaler,
            X_train,
            y_train,
            X_test,
            y_test,
            inverted_y_pred,
            rmse,
            mape,
            features_df,
            inverted_y_test,
            close_scale,
            close_min,
            split,
            scaled_data,
        ) = train_model(btc, window, epochs)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)
    col1.metric("Test RMSE", f"{rmse:.2f}")
    col2.metric("Test MAPE", f"{mape:.2f}%")

    # Predict next day
    last_features = features_df.tail(window).values
    scaled_last = scaler.transform(last_features)
    predicted_scaled = model(torch.FloatTensor(scaled_last).unsqueeze(0))
    predicted_price = (predicted_scaled.item() / close_scale) + close_min

    sentiment = get_sentiment_score()
    adjusted_price = predicted_price * (1 + 0.1 * sentiment)  # Adjust by sentiment

    st.subheader("Next Day BTC Price Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Predicted Price", f"${predicted_price:,.2f}")
    col2.metric("Sentiment Score", f"{sentiment:.2f}")
    col3.metric("Adjusted Price", f"${adjusted_price:,.2f}")

    # Plot actual vs predicted
    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(
            x=np.arange(len(inverted_y_test)),
            y=inverted_y_test,
            mode="lines",
            name="Actual",
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=np.arange(len(inverted_y_pred)),
            y=inverted_y_pred,
            mode="lines",
            name="Predicted",
        )
    )
    fig_pred.update_layout(
        title="Test Set: Actual vs Predicted",
        xaxis_title="Sample",
        yaxis_title="Price (USD)",
        template="plotly_white",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Backtesting
    sharpe, total_return = backtest(features_df, inverted_y_pred, len(y_test))
    st.subheader("Backtesting Results")
    col1, col2 = st.columns(2)
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Total Return", f"{total_return * 100:.2f}%")


if __name__ == "__main__":
    from market_watch.utils import set_page_config_once

    set_page_config_once()
    main()

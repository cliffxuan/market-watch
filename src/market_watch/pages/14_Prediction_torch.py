import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from torch import nn


# For sentiment: Use Crypto Fear & Greed Index from alternative.me
def get_fear_and_greed_index() -> int:
    try:
        response = requests.get("https://api.alternative.me/fng/")
        response.raise_for_status()
        data = response.json()
        # Value is 0-100
        return int(data["data"][0]["value"])
    except Exception as e:
        st.warning(f"Could not fetch Fear & Greed Index: {e}. Using neutral 50.")
        return 50


def get_sentiment_score():
    fng_value = get_fear_and_greed_index()
    # Normalize 0-100 to -1 to 1
    # 0 (Extreme Fear) -> -1
    # 50 (Neutral) -> 0
    # 100 (Extreme Greed) -> 1
    normalized_score = (fng_value - 50) / 50
    return normalized_score, fng_value


def prepare_data(data, window=30, test_size=0.2):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    x_inputs, y = [], []
    for i in range(len(scaled_data) - window):
        x_inputs.append(scaled_data[i : i + window])
        y.append(scaled_data[i + window, 0])  # Predict close price
    x_inputs, y = np.array(x_inputs), np.array(y)
    split = int(len(x_inputs) * (1 - test_size))
    x_train, x_test = (
        torch.FloatTensor(x_inputs[:split]),
        torch.FloatTensor(x_inputs[split:]),
    )
    y_train, y_test = torch.FloatTensor(y[:split]), torch.FloatTensor(y[split:])
    return x_train, y_train, x_test, y_test, scaler, scaled_data, split


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
    x_train, y_train, x_test, y_test, scaler, scaled_data, split = prepare_data(
        features, window
    )
    model = LSTMModel(input_size=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Evaluate on original scale
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test).numpy().flatten()
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
        x_train,
        y_train,
        x_test,
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
        "Leveraging deep learning (LSTM) for sequence modeling and LLM-driven sentiment"
        " from X for adjusted predictions. Includes backtesting for trading strategies."
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
    st.plotly_chart(fig_hist, width="stretch")

    # Train model
    try:
        (
            model,
            scaler,
            x_train,
            y_train,
            x_test,
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

    sentiment, fng_value = get_sentiment_score()
    adjusted_price = predicted_price * (1 + 0.1 * sentiment)  # Adjust by sentiment

    st.subheader("Next Day BTC Price Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Predicted Price", f"${predicted_price:,.2f}")
    col2.metric("Fear & Greed Index", f"{fng_value} ({sentiment:.2f})")
    col3.metric("Adjusted Price", f"${adjusted_price:,.2f}")

    # Plot actual vs predicted
    test_dates = features_df.index[split + window :]
    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(
            x=test_dates,
            y=inverted_y_test,
            mode="lines",
            name="Actual",
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=test_dates,
            y=inverted_y_pred,
            mode="lines",
            name="Predicted",
        )
    )
    fig_pred.update_layout(
        title="Test Set: Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
    )
    st.plotly_chart(fig_pred, width="stretch")

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

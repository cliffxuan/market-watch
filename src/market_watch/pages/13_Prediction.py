import numpy as np
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

from market_watch.utils import set_page_config_once


def prepare_data(data, window=5):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


def train_model(btc):
    close_prices = btc["Close"].values
    if len(close_prices) < 6:
        raise ValueError("❌ Not enough data to train model (need at least 6 rows).")

    X, y = prepare_data(close_prices)
    if X.size == 0:
        raise ValueError("❌ X is empty. prepare_data() produced no samples.")

    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


def load_data():
    btc = yf.Ticker("BTC-USD").history(period="10y")
    if btc is None or btc.empty:
        raise ValueError("❌ Failed to download BTC-USD data. DataFrame is empty.")
    btc = btc[["Close"]].dropna()
    return btc


def main():
    st.title("📈 Bitcoin Price Prediction Dashboard")

    # Load data
    btc = load_data()
    st.subheader("Historical BTC Prices")
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=btc.index,
            y=btc["Close"],
            mode="lines",
            name="Close",
        )
    )
    fig_hist.update_layout(
        title="BTC-USD Close Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Train model
    try:
        model, X, y = train_model(btc)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    # Predict next day
    last_window = btc["Close"].values[-5:]
    predicted_price = model.predict([last_window])[0]

    st.subheader("Next Day BTC Price Prediction")
    st.metric(label="Predicted Close Price", value=f"${predicted_price:,.2f}")

    # Plot actual vs predicted (training fit)
    y_pred = model.predict(X)
    x_idx = np.arange(len(y))
    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(
            x=x_idx,
            y=y,
            mode="lines",
            name="Actual",
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=x_idx,
            y=y_pred,
            mode="lines",
            name="Predicted",
        )
    )
    fig_pred.update_layout(
        title="Model Fit: Actual vs Predicted",
        xaxis_title="Sample",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig_pred, use_container_width=True)


if __name__ == "__main__":
    set_page_config_once()
    main()

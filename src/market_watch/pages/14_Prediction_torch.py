import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from torch import nn


def fetch_historical_fng(limit: int = 10000) -> pd.DataFrame:
    """Fetches historical Fear & Greed Index data."""
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["fng_value"] = df["value"].astype(int)
        df = df.set_index("timestamp").sort_index()
        # Keep only fng_value
        return df[["fng_value"]]
    except Exception as e:
        st.warning(f"Could not fetch historical Fear & Greed Index: {e}")
        return pd.DataFrame()


def create_sequences(dataset: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x_inputs, y = [], []
    for i in range(len(dataset) - window):
        x_inputs.append(dataset[i : i + window])
        y.append(dataset[i + window, 0])  # Predict close price
    return np.array(x_inputs), np.array(y)


def prepare_data(
    data: np.ndarray, window: int = 30, test_size: float = 0.2
) -> tuple[
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    MinMaxScaler,
    np.ndarray,
    int,
]:
    # Split raw data first to avoid leakage
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    # Overlap window for test set to ensure continuity
    test_data = data[split_idx - window :]

    scaler = MinMaxScaler()
    # Fit ONLY on training data
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    x_train, y_train = create_sequences(train_scaled, window)
    x_test, y_test = create_sequences(test_scaled, window)

    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    # Reconstruct full scaled data for consistency (though strictly only scaler matters)
    scaled_data = np.concatenate((train_scaled, test_scaled[window:]), axis=0)
    split = len(x_train)

    return x_train, y_train, x_test, y_test, scaler, scaled_data, split


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 50,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):  # input_size for features
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def train_model(
    btc: pd.DataFrame, window: int = 30, epochs: int = 50, lr: float = 0.001
) -> tuple[
    LSTMModel,
    MinMaxScaler,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    np.ndarray,
    float,
    float,
    pd.DataFrame,
    np.ndarray,
    float,
    float,
    int,
    np.ndarray,
]:
    features_df = add_all_ta_features(
        btc,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )
    # Feature Engineering for Stationarity
    # 1. Log Return (Target)
    features_df["Log_Ret"] = np.log(
        features_df["Close"] / features_df["Close"].shift(1)
    )

    # 2. Distance from SMA (Stationary proxy for trend)
    # Log difference between Close and SMA
    features_df["dist_sma_fast"] = np.log(
        features_df["Close"] / features_df["trend_sma_fast"]
    )

    # Drop NaNs created by shift/TA but keep all columns for now
    features_df = features_df.dropna()

    # Select features for training
    feature_cols = [
        "Log_Ret",
        "dist_sma_fast",
        "momentum_rsi",
        "volume_obv",
        "fng_value",
    ]

    print(f"DEBUG: Features DF Head:\n{features_df[feature_cols].head()}")
    print(f"DEBUG: Features DF Describe:\n{features_df[feature_cols].describe()}")

    features = features_df[feature_cols].values
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

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test).numpy().flatten()
        # y_test_np = y_test.numpy()

        # Inverse scale to get predicted Log Returns
        # scaler column 0 is Log_Ret
        ret_scale = scaler.scale_[0]
        ret_min = scaler.min_[0]

        pred_log_ret = (y_pred_test - ret_min) / ret_scale
        # actual_log_ret = (y_test_np - ret_min) / ret_scale

        # Reconstruct Prices
        # We need the Close price just before the test predictions start
        # Test set starts at index: split + window
        # Reconstruct Prices using One-Step-Ahead Prediction
        # Price_t = Actual_Price_{t-1} * exp(Pred_Log_Ret_t)
        # This aligns predictions with the actual price level at each step.

        # Get actual Close prices from features_df
        # The test set corresponds to features_df indices [split + window : ]
        start_idx = split + window

        # Actual prices for the test set
        actual_prices = features_df["Close"].values[start_idx:]

        # Previous actual prices (shifted by 1)
        # We need prices from [start_idx - 1] to [end - 1]
        prev_actual_prices = features_df["Close"].values[start_idx - 1 : -1]

        # Predicted Prices (One-Step-Ahead)
        pred_prices = prev_actual_prices * np.exp(pred_log_ret)

        rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
        mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100

    return (
        model,
        scaler,
        x_train,
        y_train,
        x_test,
        y_test,
        pred_prices,  # Return reconstructed prices
        rmse,
        mape,
        features_df,
        actual_prices,  # Return reconstructed actuals
        ret_scale,
        ret_min,
        split,
        scaled_data,
    )


def backtest(
    features_df: pd.DataFrame, inverted_y_pred: np.ndarray, y_test_len: int
) -> tuple[float, float]:
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


def predict_future(
    model: LSTMModel,
    scaler: MinMaxScaler,
    btc_data: pd.DataFrame,
    window: int,
    days: int = 365,
) -> pd.DataFrame:
    """
    Autoregressive prediction for future days.
    """
    future_btc = btc_data.copy()

    # We need to ensure we have the scaler fitted on the correct columns if we were
    # to inverse transform features. But here we only care about the target (Close)
    # which is column 0 in our specific scaler logic if we used it that way.
    # Wait, the scaler in train_model was fitted on 'train_data' which was just
    # the features array. We need to be careful. The model expects scaled features.

    # Let's look at how features are constructed in train_model:
    # features_df = ... [
    #     ["Close", "trend_sma_fast", "momentum_rsi", "volume_obv", "fng_value"]
    # ]
    # scaler.fit(train_data) -> train_data is these 5 columns.

    # So to predict, we need to:
    # 1. Get the last 'window' rows of features from future_btc
    # 2. Scale them
    # 3. Predict
    # 4. Inverse scale the prediction (using column 0 of scaler)
    # 5. Append new row to future_btc
    # 6. Re-calculate features

    # Progress bar
    progress_bar = st.progress(0)

    for i in range(days):
        # 1. Calculate features on current data
        # Optimization: Only use the last 200 rows to calculate TA.
        # Most indicators (RSI, SMA) need limited history.
        # This prevents the dataframe from growing indefinitely in the TA calculation.

        ta_window_size = 200
        if len(future_btc) > ta_window_size:
            calc_df = future_btc.tail(ta_window_size).copy()
        else:
            calc_df = future_btc.copy()

        features_df = add_all_ta_features(
            calc_df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True,
        )
        features_df = features_df[
            ["Close", "trend_sma_fast", "momentum_rsi", "volume_obv", "fng_value"]
        ].fillna(0)

        # Feature Engineering for Stationarity (Same as training)
        # We need the previous row to calculate Log_Ret for the current last row?
        # Actually, we just need to construct the features for the *last window*.
        # The 'Close' column in features_df is raw price.

        # Calculate Log_Ret and dist_sma_fast
        features_df["Log_Ret"] = np.log(
            features_df["Close"] / features_df["Close"].shift(1)  # type: ignore
        )
        features_df["dist_sma_fast"] = np.log(
            features_df["Close"] / features_df["trend_sma_fast"]
        )

        # Drop NaNs created by shift/TA
        features_df = features_df[
            ["Log_Ret", "dist_sma_fast", "momentum_rsi", "volume_obv", "fng_value"]
        ].dropna()  # type: ignore

        # Get last window
        last_features = features_df.tail(window).values

        # Scale
        scaled_last = scaler.transform(last_features)

        # Predict
        with torch.no_grad():
            model.eval()
            pred_scaled = model(torch.FloatTensor(scaled_last).unsqueeze(0))

        # Inverse scale (Column 0 is Log_Ret)
        pred_val_scaled = pred_scaled.item()

        pred_log_ret = (pred_val_scaled - scaler.min_[0]) / scaler.scale_[0]

        # Calculate New Price
        # New_Price = Last_Price * exp(pred_log_ret)
        last_price = future_btc["Close"].iloc[-1]
        pred_price = last_price * np.exp(pred_log_ret)

        # Append new row
        last_date = future_btc.index[-1]
        next_date = last_date + pd.Timedelta(days=1)

        # Assumptions for new row:
        # Open/High/Low = Predicted Close (neutral assumption)
        # Volume = Last Volume
        # F&G = Last F&G

        new_row = pd.DataFrame(
            {
                "Open": [pred_price],
                "High": [pred_price],
                "Low": [pred_price],
                "Close": [pred_price],
                "Volume": [future_btc["Volume"].iloc[-1]],
                "fng_value": [future_btc["fng_value"].iloc[-1]],
            },
            index=[next_date],  # type: ignore
        )

        future_btc = pd.concat([future_btc, new_row])

        if i % 10 == 0:
            progress_bar.progress((i + 1) / days)

    progress_bar.empty()
    return future_btc.iloc[-days:]


def load_data(period: str = "10y") -> pd.DataFrame:
    btc = yf.Ticker("BTC-USD").history(period=period)
    if btc.empty:
        raise ValueError("❌ Failed to download BTC-USD data.")

    # Ensure timezone naivety for merging
    if isinstance(btc.index, pd.DatetimeIndex):
        btc.index = btc.index.tz_localize(None)

    # Fetch and merge Fear & Greed data
    fng = fetch_historical_fng()
    if not fng.empty:
        # Merge on index (date)
        # F&G data is daily, BTC is daily.
        # We use 'left' join to keep all BTC rows, filling missing F&G with ffill
        btc = btc.join(fng, how="left")
        btc["fng_value"] = (
            btc["fng_value"].ffill().fillna(50)
        )  # Default to 50 if missing
    else:
        btc["fng_value"] = 50

    return btc


def main() -> None:
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

    st.subheader("Historical Fear and Greed Index")
    st.plotly_chart(px.line(btc, x=btc.index, y="fng_value"))
    # Train model
    try:
        (
            model,
            scaler,
            _,
            _,
            _,
            y_test,
            inverted_y_pred,
            rmse,
            mape,
            features_df,
            inverted_y_test,
            ret_scale,
            ret_min,
            split,
            _,
        ) = train_model(btc, window, epochs)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)
    col1.metric("Test RMSE", f"{rmse:.2f}")
    col2.metric("Test MAPE", f"{mape:.2f}%")

    # Predict next day
    feature_cols = [
        "Log_Ret",
        "dist_sma_fast",
        "momentum_rsi",
        "volume_obv",
        "fng_value",
    ]
    last_features = features_df[feature_cols].tail(window).values
    scaled_last = scaler.transform(last_features)
    predicted_scaled = model(torch.FloatTensor(scaled_last).unsqueeze(0))

    # Inverse scale Log Return
    pred_log_ret = (predicted_scaled.item() - ret_min) / ret_scale

    # Apply to last known price
    last_known_price = btc["Close"].iloc[-1]
    predicted_price = last_known_price * np.exp(pred_log_ret)

    st.subheader("Next Day BTC Price Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Price", f"${predicted_price:,.2f}")

    # Show F&G value used for prediction
    last_fng = features_df["fng_value"].iloc[-1]
    col2.metric("Current Fear & Greed", f"{int(last_fng)}")

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
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Total Return", f"{total_return * 100:.2f}%")

    # Future Prediction
    st.subheader("Future Prediction")
    st.markdown(
        "Predict prices into the future (Autoregressive). "
        "**Warning: Highly speculative.**"
    )

    forecast_days = st.slider("Days to Predict", 30, 365, 365)

    if st.button("Predict Future"):
        with st.spinner(f"Generating {forecast_days} days of predictions..."):
            # We need the original btc dataframe with OHLCV for the loop
            # 'btc' variable from main scope has it.
            future_df = predict_future(model, scaler, btc, window, days=forecast_days)

            st.success("Prediction complete!")

            # Plot
            fig_future = go.Figure()
            # Historical (last 90 days for context)
            fig_future.add_trace(
                go.Scatter(
                    x=btc.index[-90:],
                    y=btc["Close"].iloc[-90:],
                    mode="lines",
                    name="Historical (Last 90 days)",
                    line={"color": "gray"},
                )
            )
            # Future
            fig_future.add_trace(
                go.Scatter(
                    x=future_df.index,
                    y=future_df["Close"],
                    mode="lines",
                    name="Forecast",
                    line={"color": "blue", "dash": "dash"},
                )
            )
            fig_future.update_layout(
                title=f"Bitcoin Price Forecast ({forecast_days} days)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white",
            )
            st.plotly_chart(fig_future, width="stretch")


if __name__ == "__main__":
    from market_watch.utils import set_page_config_once

    set_page_config_once()
    main()

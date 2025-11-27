import random
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import yfinance as yf
from loguru import logger
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        return pd.DataFrame(df["fng_value"])
    except Exception as e:
        st.warning(f"Could not fetch historical Fear & Greed Index: {e}")
        return pd.DataFrame()


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators ensuring no look-ahead bias.
    All calculations use only past data.
    """
    df = df.copy()

    # Validate required columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns. Need: {required}")

    # Price-based indicators (naturally backward-looking)
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI (14-period) - proper calculation
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)  # Avoid division by zero
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    bb_middle = df["Close"].rolling(window=20, min_periods=20).mean()
    bb_std = df["Close"].rolling(window=20, min_periods=20).std()
    df["BB_Upper"] = bb_middle + (bb_std * 2)
    df["BB_Lower"] = bb_middle - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_middle
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
        df["BB_Upper"] - df["BB_Lower"]
    )

    # Volume indicators
    df["Volume_SMA"] = df["Volume"].rolling(window=20, min_periods=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"].replace(0, np.nan)

    return df


def create_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stationary features ensuring no data leakage.
    Returns (percentage change) are naturally backward-looking.
    """
    df = df.copy()

    # Returns (these naturally use past prices)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Price_Change"] = df["Close"].pct_change()

    # Volatility (rolling standard deviation of returns)
    df["Volatility"] = df["Log_Return"].rolling(window=20, min_periods=20).std()

    # Momentum (comparing current to past prices)
    df["Momentum_5"] = df["Close"].pct_change(periods=5)
    df["Momentum_10"] = df["Close"].pct_change(periods=10)

    # Distance from moving averages
    if "SMA_20" in df.columns:
        df["Dist_SMA_20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"].replace(
            0, np.nan
        )
    if "EMA_20" in df.columns:
        df["Dist_EMA_20"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"].replace(
            0, np.nan
        )

    # Drop first row (NaN from pct_change)
    return df.iloc[1:].copy()


def create_sequences(dataset: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x_inputs, y = [], []
    for i in range(len(dataset) - window):
        x_inputs.append(dataset[i : i + window])
        y.append(dataset[i + window, 0])  # Predict close price
    return np.array(x_inputs), np.array(y)


def prepare_data_with_validation(
    data: np.ndarray,
    window: int = 30,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    MinMaxScaler,
    int,
]:
    """Time-series aware split with validation set"""
    n = len(data)
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - val_size))

    # Train: 70%, Validation: 15%, Test: 15%
    train_data = data[:val_start]
    val_data = data[val_start - window : test_start]
    test_data = data[test_start - window :]

    scaler = MinMaxScaler()
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    x_train, y_train = create_sequences(train_scaled, window)
    x_val, y_val = create_sequences(val_scaled, window)
    x_test, y_test = create_sequences(test_scaled, window)

    return (
        torch.FloatTensor(x_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(x_val),
        torch.FloatTensor(y_val),
        torch.FloatTensor(x_test),
        torch.FloatTensor(y_test),
        scaler,
        test_start,
    )


def get_device(device_preference: str = "Auto") -> torch.device:
    """
    Get device based on preference and availability.
    """
    if device_preference in ["CPU", "CUDA", "MPS"]:
        logger.info(f"Using {device_preference} (Forced)")
        return torch.device(device_preference.lower())
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def get_available_devices() -> list[str]:
    """Get list of available compute devices."""
    devices = ["Auto", "CPU"]
    if torch.cuda.is_available():
        devices.append("CUDA")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("MPS")
    return devices


class BTCPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.batch_norm = nn.BatchNorm1d(lstm_output_size)
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        num_directions = 2 if self.lstm.bidirectional else 1

        h0 = torch.zeros(
            self.lstm.num_layers * num_directions,
            batch_size,
            self.lstm.hidden_size,
        ).to(x.device)
        c0 = torch.zeros(
            self.lstm.num_layers * num_directions,
            batch_size,
            self.lstm.hidden_size,
        ).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        return out


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad()
    device = next(model.parameters()).device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    outputs = model(x_train)
    loss = criterion(outputs, y_train.unsqueeze(1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def validate_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        outputs = model(x_val)
        loss = criterion(outputs, y_val.unsqueeze(1))
    return loss.item()


def select_features(btc_stationary: pd.DataFrame) -> list[str]:
    # Select final features - using only available columns
    possible_features = [
        "Log_Return",
        "Price_Change",
        "Volatility",
        "RSI",
        "MACD",
        "BB_Position",
        "BB_Width",
        "Momentum_5",
        "Momentum_10",
        "Dist_SMA_20",
        "Dist_EMA_20",
        "Volume_Ratio",
        "fng_value",
    ]

    # Ensure we have all columns
    available_cols = [col for col in possible_features if col in btc_stationary.columns]

    # Add essential features if missing
    if "Log_Return" not in available_cols:
        btc_stationary["Log_Return"] = np.log(
            btc_stationary["Close"] / btc_stationary["Close"].shift(1)
        )
        available_cols.append("Log_Return")

    if "fng_value" not in available_cols and "fng_value" in btc_stationary.columns:
        available_cols.append("fng_value")

    # Ensure we have at least 3 features
    min_features = 3
    if len(available_cols) < min_features:
        # Add some basic features
        if "Price_Change" not in available_cols:
            btc_stationary["Price_Change"] = btc_stationary["Close"].pct_change()
            available_cols.append("Price_Change")
        if "Volume_Ratio" not in available_cols:
            btc_stationary["Volume_Ratio"] = (
                btc_stationary["Volume"] / btc_stationary["Volume"].rolling(20).mean()
            )
            available_cols.append("Volume_Ratio")

    return available_cols


# @st.cache_resource
def train_model(  # noqa: C901
    btc: pd.DataFrame,
    window: int = 30,
    epochs: int = 100,
    lr: float = 0.001,
    device_name: str = "Auto",
) -> tuple[
    BTCPredictor,
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
    list[float],
    list[float],
    list[str],
]:
    # Ensure we have the required columns
    required_columns = ["Open", "High", "Low", "Close", "Volume", "fng_value"]
    missing_columns = [col for col in required_columns if col not in btc.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Calculate technical indicators
    try:
        btc_tech = calculate_technical_indicators(btc)
    except Exception as e:
        st.warning(
            f"Technical indicators calculation failed: {e}. Using basic features."
        )
        btc_tech = btc.copy()

    # Create stationary features
    try:
        btc_stationary = create_stationary_features(btc_tech)
    except Exception as e:
        st.warning(f"Stationary features creation failed: {e}. Using basic features.")
        btc_stationary = btc_tech.copy()
        # Add basic features manually
        btc_stationary["Log_Return"] = np.log(
            btc_stationary["Close"] / btc_stationary["Close"].shift(1)
        )
        btc_stationary["Price_Change"] = btc_stationary["Close"].pct_change()
        btc_stationary["Volume_Ratio"] = (
            btc_stationary["Volume"] / btc_stationary["Volume"].rolling(20).mean()
        )
        btc_stationary = btc_stationary.dropna()

    available_cols = select_features(btc_stationary)
    print(f"Using features: {available_cols}")

    # Handle NaN values
    btc_stationary = (
        btc_stationary[[*available_cols, "Close"]].fillna(method="ffill").fillna(0)  # type: ignore
    )
    features = btc_stationary[available_cols].to_numpy()

    if len(features) < window + 1:
        raise ValueError(f"❌ Not enough data (need at least {window + 1} rows).")

    # Prepare data with proper time series split
    x_train, y_train, x_val, y_val, x_test, y_test, scaler, test_start = (
        prepare_data_with_validation(features.astype(np.float32), window)
    )

    model = BTCPredictor(input_size=len(available_cols)).to(get_device(device_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, x_train, y_train)
        val_loss = validate_one_epoch(model, criterion, x_val, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        y_pred_test = model(x_test.to(device)).cpu().numpy().flatten()

    # Reconstruct prices
    ret_scale = scaler.scale_[0]
    ret_min = scaler.min_[0]

    pred_log_ret = (y_pred_test - ret_min) / ret_scale

    # Get actual prices for the test period
    # Ensure we have enough data
    end_idx = min(test_start + len(y_pred_test), len(btc_stationary))
    actual_prices = btc_stationary.iloc[test_start:end_idx]["Close"].values

    # Previous actual prices (for price reconstruction)
    prev_start = max(0, test_start - 1)
    prev_end = min(test_start - 1 + len(y_pred_test), len(btc_stationary))
    prev_actual_prices = btc_stationary.iloc[prev_start:prev_end]["Close"].values

    # Ensure same length
    min_len = min(len(actual_prices), len(prev_actual_prices), len(pred_log_ret))
    actual_prices = actual_prices[:min_len]
    prev_actual_prices = prev_actual_prices[:min_len]
    pred_log_ret = pred_log_ret[:min_len]

    # Predicted Prices (One-Step-Ahead)
    pred_prices = prev_actual_prices * np.exp(pred_log_ret)

    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100

    return (
        model,
        scaler,
        x_train,
        y_train,
        x_val,
        y_val,
        pred_prices,
        rmse,
        mape,
        btc_stationary,
        actual_prices,
        ret_scale,
        ret_min,
        test_start,
        train_losses,
        val_losses,
        available_cols,
    )


def backtest(
    features_df: pd.DataFrame,
    inverted_y_pred: np.ndarray,
    test_start: int,
    y_test_len: int,
) -> tuple[float, float]:
    """Improved backtesting with proper indexing"""
    try:
        # Ensure features_df has Close column
        if "Close" not in features_df.columns:
            return 0.0, 0.0

        # Use the test period for backtesting
        end_idx = min(test_start + y_test_len, len(features_df))
        test_prices = features_df["Close"].iloc[test_start:end_idx]

        prev_start = max(0, test_start - 1)
        prev_end = min(test_start - 1 + y_test_len, len(features_df))
        test_prev_prices = features_df["Close"].iloc[prev_start:prev_end]

        # Ensure same length
        min_len = min(len(test_prices), len(test_prev_prices), len(inverted_y_pred))
        test_prices = test_prices[:min_len]
        test_prev_prices = test_prev_prices[:min_len]
        inverted_y_pred = inverted_y_pred[:min_len]

        # Generate signals based on predicted vs current price
        signals = np.sign(inverted_y_pred - test_prev_prices.values)

        # Calculate returns
        pct_returns = test_prices.pct_change().fillna(0).values

        # Ensure signals and returns have same length
        min_len = min(len(signals), len(pct_returns))
        signals = signals[:min_len]
        pct_returns = pct_returns[:min_len]

        returns = signals * pct_returns
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0, 0.0

        cumulative = np.cumprod(1 + returns) - 1
        sharpe = (
            (np.mean(returns) / np.std(returns) * np.sqrt(252))
            if np.std(returns) > 0
            else 0
        )
        total_return = cumulative[-1] if len(cumulative) > 0 else 0
        return sharpe, total_return
    except Exception as e:
        print(f"Backtesting error: {e}")
        return 0.0, 0.0


def robust_future_prediction(
    model: BTCPredictor,
    scaler: MinMaxScaler,
    btc_data: pd.DataFrame,
    window: int,
    feature_cols: list[str],
    days: int = 60,
) -> pd.DataFrame:
    """
    Future prediction with dampened model output to prevent explosion.
    Uses the model's predicted log returns but applies a decay factor over time.
    """
    current_price = btc_data["Close"].iloc[-1]
    future_predictions = []
    future_dates = []

    progress_bar = st.progress(0)

    # Calculate historical statistics for realistic bounds
    returns = btc_data["Close"].pct_change().dropna()
    daily_volatility = returns.std()

    st.write(f"Historical daily volatility: {daily_volatility * 100:.2f}%")

    # Start with the last window of data
    # We need to reconstruct the feature matrix for the last window
    # logic similar to train_model but just for the last window
    try:
        # We'll reuse the logic from train_model to get the last window's
        # features. But since we don't have a clean way to call it, we'll approximate.
        # But since we don't have a clean way to call it, we'll approximate.
        # Ideally, we should have a 'get_features' function.
        # For now, we'll use the passed feature_cols and assume we can construct them.
        # Actually, passing the last known feature vector is safer.
        # But we need to update it.

        # Simplified approach:
        # 1. Get last known features from the dataframe used in training (if possible)
        #    But we only have btc_data (raw).
        #    We need to re-calculate features for btc_data.
        btc_tech = calculate_technical_indicators(btc_data)
        btc_stationary = create_stationary_features(btc_tech)

        # Ensure we have all needed columns
        for col in feature_cols:
            if col not in btc_stationary.columns:
                btc_stationary[col] = 0.0  # Fallback

        # Get the last window of features
        last_features = btc_stationary[feature_cols].tail(window).values

        if len(last_features) < window:
            raise ValueError("Not enough history for prediction window")

        current_features = last_features.copy()
        current_prediction = current_price

        for day in range(days):
            # Scale
            scaled_features = scaler.transform(current_features)

            # Predict
            with torch.no_grad():
                model.eval()
                device = next(model.parameters()).device
                input_tensor = (
                    torch.FloatTensor(scaled_features).unsqueeze(0).to(device)
                )
                pred_scaled = model(input_tensor)

            # Inverse scale
            # Log_Ret is always index 0 in our scaler logic?
            # Wait, scaler was fitted on 'available_cols'.
            # We need to know which index is 'Log_Return'.
            try:
                log_ret_idx = feature_cols.index("Log_Return")
            except ValueError:
                log_ret_idx = 0  # Fallback assumption

            ret_scale = scaler.scale_[log_ret_idx]
            ret_min = scaler.min_[log_ret_idx]

            raw_pred_log_ret = (pred_scaled.item() - ret_min) / ret_scale

            # Apply DAMPENING
            # As we go further into the future, we trust the model less.
            # Decay factor: 0.95^day
            decay = 0.95**day
            dampened_log_ret = raw_pred_log_ret * decay

            # Apply volatility constraint
            # Clip to +/- 2 standard deviations of history
            max_move = daily_volatility * 2
            dampened_log_ret = np.clip(dampened_log_ret, -max_move, max_move)

            # Calculate new price
            new_price = current_prediction * np.exp(dampened_log_ret)

            # Hard bounds (sanity check)
            new_price = max(new_price, current_price * 0.5)
            new_price = min(new_price, current_price * 2.0)

            future_predictions.append(new_price)
            future_dates.append(btc_data.index[-1] + pd.Timedelta(days=day + 1))

            # Update state for next step
            current_prediction = new_price

            # Update features (Simplified autoregression)
            # Shift window
            new_row = current_features[-1].copy()

            # Update Log_Return in the new row
            new_row[log_ret_idx] = dampened_log_ret

            # Update other features if possible (e.g. Price_Change)
            if "Price_Change" in feature_cols:
                idx = feature_cols.index("Price_Change")
                new_row[idx] = np.exp(dampened_log_ret) - 1

            # Append new row and drop first
            # Append new row and drop first
            current_features = np.vstack(
                [current_features[1:], new_row.astype(np.float32)]
            )

            if day % 10 == 0:
                progress_bar.progress((day + 1) / days)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return pd.DataFrame()

    progress_bar.empty()

    result_df = pd.DataFrame(
        {"Date": future_dates, "Predicted_Close": future_predictions}
    ).set_index("Date")

    return result_df


@st.cache_data(ttl="1h")
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
        btc = btc.join(fng, how="left")
        btc["fng_value"] = btc["fng_value"].ffill().fillna(50)
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
    with st.sidebar:
        st.header("Configuration")
        period = st.selectbox("Data Period", ["1y", "5y", "10y", "max"], index=2)
        window = st.slider("Sequence Window", 10, 60, 30)
        epochs = st.slider("Training Epochs", 20, 200, 100)
        device_pref = st.selectbox(
            "Compute Device",
            get_available_devices(),
            index=0,
            help="Force CPU for reproducible results. MPS/CUDA is faster but may vary.",
        )

    set_seed(42)

    # Load data
    try:
        btc = load_data(period)
        st.success(f"✅ Loaded {len(btc)} days of BTC data")
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    # Display data info
    st.subheader("Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Start Date", pd.to_datetime(btc.index[0]).strftime("%Y-%m-%d"))
    col2.metric("End Date", pd.to_datetime(btc.index[-1]).strftime("%Y-%m-%d"))
    col3.metric("Current Price", f"${btc['Close'].iloc[-1]:.2f}")
    col4.metric("F&G Index", f"{btc['fng_value'].iloc[-1]}")

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

    st.subheader("Historical Fear and Greed Index")
    fig_fng = px.line(btc, x=btc.index, y="fng_value", title="Fear & Greed Index")
    fig_fng.update_layout(template="plotly_white")
    st.plotly_chart(fig_fng, use_container_width=True)

    # Train model
    st.subheader("Model Training")

    # Check if we need to retrain
    # We retrain if:
    # 1. Model is not in session state
    # 2. Parameters have changed
    current_params = {
        "period": period,
        "window": window,
        "epochs": epochs,
        "device": device_pref,
    }

    if st.session_state.get("training_params") != current_params:
        with st.spinner("Training model... This may take a few minutes."):
            try:
                results = train_model(btc, window, epochs, device_name=device_pref)
                st.session_state["training_results"] = results
                st.session_state["training_params"] = current_params
                st.success("✅ Model training completed successfully!")
            except Exception as e:
                st.error(f"Model training failed: {e}")
                st.stop()
    else:
        st.info("Using cached model from session state.")

    # Unpack results from session state
    (
        model,
        scaler,
        _,
        _,
        _,
        _,
        inverted_y_pred,
        rmse,
        mape,
        features_df,
        inverted_y_test,
        ret_scale,
        ret_min,
        test_start,
        train_losses,
        val_losses,
        feature_cols,
    ) = st.session_state["training_results"]

    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)
    col1.metric("Test RMSE", f"{rmse:.2f}")
    col2.metric("Test MAPE", f"{mape:.2f}%")

    # Plot training history
    if train_losses and val_losses:
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(y=train_losses, mode="lines", name="Training Loss")
        )
        fig_loss.add_trace(
            go.Scatter(y=val_losses, mode="lines", name="Validation Loss")
        )
        fig_loss.update_layout(
            title="Training History",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Predict next day
    try:
        last_features = features_df[feature_cols].tail(window).values
        scaled_last = scaler.transform(last_features)

        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            input_tensor = torch.FloatTensor(scaled_last).unsqueeze(0).to(device)
            predicted_scaled = model(input_tensor)

        # Inverse scale Log Return
        pred_log_ret = (predicted_scaled.item() - ret_min) / ret_scale

        # Apply to last known price
        last_known_price = btc["Close"].iloc[-1]
        predicted_price = last_known_price * np.exp(pred_log_ret)

        st.subheader("Next Day BTC Price Prediction")
        col1, col2 = st.columns(2)
        col1.metric(
            "Predicted Price",
            f"${predicted_price:,.2f}",
            delta=f"${predicted_price - last_known_price:,.2f}",
        )

        # Show F&G value used for prediction
        last_fng = features_df["fng_value"].iloc[-1]
        col2.metric("Current Fear & Greed", f"{int(last_fng)}")
    except Exception as e:
        st.warning(f"Next day prediction failed: {e}")

    # Plot actual vs predicted
    try:
        test_dates = features_df.index[test_start : test_start + len(inverted_y_test)]

        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(
                x=test_dates,
                y=inverted_y_test,
                mode="lines",
                name="Actual",
                line={"color": "blue"},
            )
        )
        fig_pred.add_trace(
            go.Scatter(
                x=test_dates,
                y=inverted_y_pred,
                mode="lines",
                name="Predicted",
                line={"color": "red", "dash": "dash"},
            )
        )
        fig_pred.update_layout(
            title="Test Set: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot predictions: {e}")

    # Backtesting
    try:
        sharpe, total_return = backtest(
            features_df, inverted_y_pred, test_start, len(inverted_y_pred)
        )
        st.subheader("Backtesting Results")
        col1, col2 = st.columns(2)
        col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col2.metric("Total Return", f"{total_return * 100:.2f}%")
    except Exception as e:
        st.warning(f"Backtesting failed: {e}")

    # Future Prediction
    st.subheader("Future Prediction")
    st.markdown(
        "Predict prices into the future (Autoregressive). "
        "**Warning: Highly speculative.**"
    )

    forecast_days = st.slider("Days to Predict", 30, 365, 60)

    # Feature selection for future prediction (optional, but good for transparency)
    # We use the same features as training.

    if st.button("Predict Future"):
        with st.spinner(f"Generating {forecast_days} days of predictions..."):
            try:
                future_df = robust_future_prediction(
                    model, scaler, btc, window, feature_cols, days=forecast_days
                )

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
                        y=future_df["Predicted_Close"],
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
                st.plotly_chart(fig_future, use_container_width=True)

                # Show prediction statistics
                st.subheader("Forecast Statistics")
                col1, col2, col3 = st.columns(3)
                current_price = btc["Close"].iloc[-1]
                predicted_end_price = future_df["Predicted_Close"].iloc[-1]
                price_change = (
                    (predicted_end_price - current_price) / current_price
                ) * 100

                col1.metric("Current Price", f"${current_price:,.2f}")
                col2.metric("Predicted End Price", f"${predicted_end_price:,.2f}")
                col3.metric("Predicted Change", f"{price_change:+.2f}%")
            except Exception as e:
                st.error(f"Future prediction failed: {e}")


if __name__ == "__main__":
    from market_watch.utils import set_page_config_once

    set_page_config_once()
    main()

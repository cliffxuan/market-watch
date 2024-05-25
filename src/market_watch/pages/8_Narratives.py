import pandas as pd
import streamlit as st
import yfinance as yf
from market_watch.utils import set_page_config_once

NARRATIVES = {
    "BTC": [
        {"name": "BTC", "ticker": "BTC-USD"},
    ],
    "SOL": [
        {"name": "PYTH", "ticker": "PYTH-USD"},
    ],
    "LAYER1": [
        {"name": "SUI", "ticker": "SUI20947-USD"},
    ],
    "RWA": [
        {"name": "ONDO", "ticker": "ONDO-USD"},
    ],
}


@st.cache_data
def get_hist(name: str) -> pd.DataFrame:
    ticker = yf.ticker.Ticker(name)
    return ticker.history(period="max")[["Open", "High", "Low", "Close", "Volume"]]  # type: ignore


def main():
    st.markdown("# Naratives")
    for narrative in NARRATIVES:
        st.markdown(f"## {narrative}")
        for coin in NARRATIVES[narrative]:
            hist = get_hist(coin["ticker"])
            st.markdown(f"### {coin['name']}")
            st.dataframe(hist)


if __name__ == "__main__":
    set_page_config_once()
    main()

import pandas as pd
import plotly.express as px
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
    "AI": [
        {"name": "RNDR", "ticker": "RNDR-USD"},
    ],
    "RWA": [
        {"name": "ONDO", "ticker": "ONDO-USD"},
        {"name": "CFG", "ticker": "CFG-USD"},
        {"name": "Pendle", "ticker": "Pendle-USD"},
        {"name": "GFI", "ticker": "GFI13967-USD"},
    ],
}


@st.cache_data
def get_hist(name: str) -> pd.DataFrame:
    ticker = yf.ticker.Ticker(name)
    return ticker.history(period="max")[["Open", "High", "Low", "Close", "Volume"]]  # type: ignore


def main():
    st.markdown("# Naratives")
    spike_setting = dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        spikecolor="grey",
    )
    for narrative in NARRATIVES:
        st.markdown(f"## {narrative}")
        for coin in NARRATIVES[narrative]:
            hist = get_hist(coin["ticker"])
            st.markdown(f"### {coin['name']}")
            st.dataframe(hist)
            fig = px.line(hist, x=hist.index, y="Close")
            fig.update_xaxes(**spike_setting)
            fig.update_yaxes(**spike_setting)
            fig.update_layout(hoverdistance=0)
            st.plotly_chart(fig)


if __name__ == "__main__":
    set_page_config_once()
    main()

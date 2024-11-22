import datetime as dt
from typing import FrozenSet

import pandas as pd
import streamlit as st

from market_watch import yahoo_finance as yf2
from market_watch.index_table import index_table
from market_watch.ticker_data import get_tickers_hists
from market_watch.utils import set_page_config_once

symbols = frozenset(
    [
        "NVDA",
        "MSTR",
        "COIN",
        "TSLA",
        "RBLX",
        "SNOW",
        "MARA",
        "CLSK",
    ]
)


@st.cache_data(ttl="1h")
def get_tickers_info(symbols: FrozenSet[str]) -> dict:
    return {
        "data": {symbol: yf2.get_info(symbol) for symbol in symbols},
        "creation_time": dt.datetime.now(tz=dt.timezone.utc),
    }


@st.cache_data(ttl="1h")
def get_close_prices(symbols: FrozenSet[str]) -> pd.DataFrame:
    return get_tickers_hists(list(symbols))["Close"]


if __name__ == "__main__":
    set_page_config_once()
    info = {}
    index_table(
        "My Watchlist",
        list(symbols),
        tickers_info=get_tickers_info(symbols),
        close_prices=get_close_prices(symbols),
    )

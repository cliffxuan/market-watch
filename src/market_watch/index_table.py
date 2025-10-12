import datetime as dt
import os
from typing import FrozenSet

import numpy as np
import pandas as pd
import streamlit as st

from market_watch import yahoo_finance as yf2
from market_watch.ticker_data import calculate_returns, get_tickers_hists
from market_watch.utils import DATA_DIR, display_tickers


def search(df: pd.DataFrame, regex: str, case: bool = False) -> pd.DataFrame:
    """Search all the columns of rows with any matches."""
    mask = np.column_stack(
        [
            df[col].astype(str).str.contains(regex, regex=True, case=case, na=False)
            for col in df
        ]
    )
    return df.loc[mask.any(axis=1)]


@st.cache_data(ttl="1h")
def get_tickers_info(symbols: FrozenSet[str]) -> dict:
    return {
        "data": {symbol: yf2.get_info(symbol) for symbol in symbols},
        "creation_time": dt.datetime.now(tz=dt.timezone.utc),
    }


@st.cache_data(ttl="1h")
def get_close_prices(symbols: FrozenSet[str]) -> pd.DataFrame:
    return get_tickers_hists(list(symbols))["Close"]


def get_info(symbols: list[str]) -> tuple[pd.DataFrame, dt.datetime]:
    tickers_info = get_tickers_info(frozenset(symbols))
    close_prices = get_close_prices(frozenset(symbols))
    return (
        calculate_returns(
            data=tickers_info["data"],
            close_prices=close_prices,
            symbols=symbols,
        ),
        tickers_info["creation_time"],
    )


def tickers_table(name: str, symbols: list[str]) -> None:
    constituents, creation_time = get_info(symbols)
    st.markdown(f"# {name}")
    st.markdown(f"Select {name} constituents to build a portfolio")
    query = st.columns(2)[0].text_input(
        "search",
        label_visibility="collapsed",
        placeholder="search",
    )
    constituents = search(constituents, query)
    cols = list(constituents)
    constituents["Select"] = False
    st.markdown(f"last updated: {creation_time.strftime('%Y-%m-%dT%H:%M%z')}")
    constituents["Market Cap"] = (constituents["Market Cap"] / 1_000_000).astype(int)
    constituents["Volume"] = (constituents["Volume"] / 1_000_000).round(1)
    constituents = st.data_editor(
        constituents,
        column_order=["Select", *cols],
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "",
                help="Select to see more info and portfolio optimization",
                default=False,
            ),
            "Market Cap": st.column_config.NumberColumn("Market Cap ($M)"),
            "Volume": st.column_config.NumberColumn("Vol ($M)"),
        },
        disabled=cols,
        hide_index=True,
        height=None if len(constituents) < 22 else 800,
    )
    symbols = list(constituents[constituents["Select"]]["Symbol"])
    if symbols:
        display_tickers(symbols, optimize=True)


def index_table(name: str, title: str) -> None:
    file = DATA_DIR / name / "latest.csv"
    constituents = pd.read_csv(file)
    creation_time = dt.datetime.fromtimestamp(os.path.getctime(file))
    st.markdown(f"# {title}")
    st.markdown("Select constituents to build a portfolio")
    query = st.columns(2)[0].text_input(
        "search",
        label_visibility="collapsed",
        placeholder="search",
    )
    constituents = search(constituents, query)
    cols = list(constituents)
    constituents["Select"] = False
    st.markdown(f"last updated: {creation_time.strftime('%Y-%m-%dT%H:%M%z')}")
    constituents["Market Cap"] = (constituents["Market Cap"] / 1_000_000).astype(int)
    constituents["Volume"] = (constituents["Volume"] / 1_000_000).round(1)
    constituents = st.data_editor(
        constituents,
        column_order=["Select", *cols],
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "",
                help="Select to see more info and portfolio optimization",
                default=False,
            ),
            "Market Cap": st.column_config.NumberColumn("Market Cap ($M)"),
            "Volume": st.column_config.NumberColumn("Vol ($M)"),
        },
        disabled=cols,
        hide_index=True,
        height=None if len(constituents) < 22 else 800,
    )
    symbols = list(constituents[constituents["Select"]]["Symbol"])
    if symbols:
        display_tickers(symbols, optimize=True)

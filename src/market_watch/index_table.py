import numpy as np
import pandas as pd
import streamlit as st

from market_watch.ticker_data import rank_by_market_cap
from market_watch.utils import (
    display_tickers,
    get_tickers_hist,
    get_tickers_info,
)


def search(df: pd.DataFrame, regex: str, case: bool = False) -> pd.DataFrame:
    """Search all the columns of rows with any matches."""
    mask = np.column_stack(
        [
            df[col].astype(str).str.contains(regex, regex=True, case=case, na=False)
            for col in df
        ]
    )
    return df.loc[mask.any(axis=1)]


def get_info(
    symbols: list[str],
    tickers_info: dict | None = None,
    close_prices: pd.DataFrame | None = None,
):
    tickers_info = tickers_info if tickers_info is not None else get_tickers_info()
    close_prices = (
        close_prices if close_prices is not None else get_tickers_hist()["Close"]
    )
    # Create DataFrame from tickers info
    constituents = pd.DataFrame.from_dict(
        {
            symbol: {
                "Symbol": symbol,
                "Name": val["price"]["shortName"],
                "Market Cap": val["price"]["marketCap"]["raw"],
                "Volume": val["summaryDetail"]["volume"]["raw"],
            }
            for symbol in symbols
            if (val := tickers_info["data"].get(symbol))
        },
        orient="index",
    )

    periods = {
        1: "1d",
        7: "7d",
        30: "30d",
        90: "90d",
        180: "6mo",
        365: "1y",
        730: "2y",
        1095: "3y",
        1460: "4y",
    }
    for period, label in periods.items():
        constituents[f"{label}%"] = (
            close_prices.iloc[-1] / close_prices.iloc[-period] * 100 - 100
        ).round(2)

    constituents = rank_by_market_cap(constituents)
    return constituents, tickers_info["creation_time"]


def index_table(
    name: str,
    symbols: list[str],
    tickers_info: dict | None = None,
    close_prices: pd.DataFrame | None = None,
) -> None:
    constituents, creation_time = get_info(symbols, tickers_info, close_prices)
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
    constituents = st.data_editor(
        constituents,
        column_order=["Select", *cols],
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "",
                help="Select to see more info and portfolio optimization",
                default=False,
            )
        },
        disabled=cols,
        hide_index=True,
        height=None if len(constituents) < 22 else 800,
    )
    symbols = list(constituents[constituents["Select"]]["Symbol"])
    if symbols:
        display_tickers(symbols, optimize=True)

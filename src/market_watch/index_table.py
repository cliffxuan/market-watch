import numpy as np
import pandas as pd
import streamlit as st

from market_watch.utils import (
    display_tickers,
    get_spx_hists,
    get_tickers_info,
)


def rank_by_market_cap(constituents: pd.DataFrame) -> pd.DataFrame:
    constituents = constituents.sort_values(
        by=["Market Cap"], ascending=False
    ).reset_index(drop=True)
    goog = constituents.Symbol.loc[lambda x: x.isin(["GOOGL", "GOOG"])]
    rank = constituents.index.map(lambda n: n + 1 if n <= goog.index.min() else n)
    constituents.insert(0, "Rank", rank)
    return constituents


def search(df: pd.DataFrame, regex: str, case: bool = False) -> pd.DataFrame:
    """Search all the columns of rows with any matches."""
    mask = np.column_stack(
        [
            df[col].astype(str).str.contains(regex, regex=True, case=case, na=False)
            for col in df
        ]
    )
    return df.loc[mask.any(axis=1)]


def index_table(name: str, symbols: list[str]) -> None:
    st.markdown(f"# {name}")
    tickers_info = get_tickers_info()

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

    close = get_spx_hists()["Close"]

    periods = {1: "1d", 7: "7d", 30: "30d", 90: "90d", 180: "6mo", 365: "1y"}
    for period, label in periods.items():
        constituents[label] = (close.iloc[-1] / close.iloc[-period] * 100 - 100).round(
            2
        )

    constituents = rank_by_market_cap(constituents)
    st.markdown(f"Select {name} constituents to build a portfolio")
    query = st.columns(2)[0].text_input(
        "search",
        label_visibility="collapsed",
        placeholder="search",
    )
    constituents = search(constituents, query)
    cols = list(constituents)
    constituents["Select"] = False
    st.markdown(f"last updated: {tickers_info['creation_time']}")
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

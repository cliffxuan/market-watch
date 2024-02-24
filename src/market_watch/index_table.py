import numpy as np
import pandas as pd
import streamlit as st

from market_watch.utils import (
    DATA_DIR,
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


def index_table(name, csv_file, cols):
    st.markdown(f"# {name}")
    constituents = pd.read_csv(DATA_DIR / csv_file)
    info = pd.DataFrame.from_dict(
        {
            key: {
                "Exchange": val["price"]["exchange"],
                "Market Cap": val["price"]["marketCap"]["raw"],
            }
            for key, val in get_tickers_info().items()
        },
        orient="index",
    )
    constituents = constituents[cols].join(info, on="Symbol")
    close = get_spx_hists()["Close"]
    constituents = constituents.join(
        (close.iloc[-1] / close.iloc[-2] * 100 - 100).round(2).to_frame("1d %"),
        on="Symbol",
    )
    constituents.insert(2, "Market Cap", constituents.pop("Market Cap"))
    constituents.insert(3, "1d %", constituents.pop("1d %"))
    constituents = constituents.join(
        (close.iloc[-1] / close.iloc[-6] * 100 - 100).round(2).to_frame("7d %"),
        on="Symbol",
    )
    constituents.insert(4, "7d %", constituents.pop("7d %"))
    constituents = rank_by_market_cap(constituents)
    st.markdown(f"Select {name} constituents to build a portfolio")
    query = st.columns(2)[0].text_input(
        "search",
        label_visibility="collapsed",
        placeholder="search",
    )
    constituents = search(constituents, query)
    cols = list(constituents)
    constituents[""] = False
    constituents = st.data_editor(
        constituents,
        column_order=["", *cols],
        disabled=cols,
        hide_index=True,
        height=None if len(constituents) < 22 else 800,
    )
    symbols = list(constituents[constituents[""]]["Symbol"])
    if symbols:
        display_tickers(symbols, optimise=True)

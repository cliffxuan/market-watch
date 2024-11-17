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

    # Calculate daily and weekly percentage changes
    constituents = (
        constituents.join(
            (close.iloc[-1] / close.iloc[-2] * 100 - 100).round(2).to_frame("1d %"),
            on="Symbol",
        )
        .join(
            (close.iloc[-1] / close.iloc[-6] * 100 - 100).round(2).to_frame("7d %"),
            on="Symbol",
        )
        .join(
            (close.iloc[-1] / close.iloc[-29] * 100 - 100).round(2).to_frame("30d %"),
            on="Symbol",
        )
    )

    # Reorder columns
    constituents.insert(2, "Market Cap", constituents.pop("Market Cap"))
    constituents.insert(3, "1d %", constituents.pop("1d %"))
    constituents.insert(4, "7d %", constituents.pop("7d %"))
    constituents.insert(5, "30d %", constituents.pop("30d %"))

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

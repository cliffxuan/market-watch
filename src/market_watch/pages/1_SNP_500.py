import json

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder

from market_watch.utils import DATA_DIR, data, set_page_config_once


def rank_by_market_cap(constituents: pd.DataFrame) -> pd.DataFrame:
    constituents = constituents.sort_values(
        by=["Market Cap"], ascending=False
    ).reset_index(drop=True)
    goog = constituents.Symbol.loc[lambda x: x.isin(["GOOGL", "GOOG"])]
    rank = constituents.index.map(lambda n: n + 1 if n <= goog.index.min() else n)
    constituents.insert(0, "Rank", rank)
    return constituents


def main():
    st.markdown("# S&P 500")
    constituents = pd.read_csv(DATA_DIR / "spx_constituents.csv")
    market_caps = []
    exchanges = []
    for symbol in constituents["Symbol"]:
        with open(DATA_DIR / "tickers" / f"{symbol}.json", "r") as f:
            info = json.loads(f.read())
            market_cap = info.get("marketCap", 0)
            market_caps.append(market_cap)
            exchange = info.get("exchange", "")
            exchanges.append(exchange)
    constituents.insert(2, "Market Cap", pd.Series(market_caps))
    constituents["Exchange"] = pd.Series(exchanges)
    constituents = rank_by_market_cap(constituents)
    builder = GridOptionsBuilder.from_dataframe(constituents)
    builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    builder.configure_column(
        "Market Cap",
        type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
        valueFormatter="Intl.NumberFormat('en', { notation: 'compact' }).format(data['Market Cap'])",
    )
    builder.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
        rowMultiSelectWithClick=True,
    )
    options = builder.build()
    aggrid = AgGrid(
        constituents,
        options,
        enable_enterprise_modules=False,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        enable_quicksearch=True,
    )
    if aggrid.selected_rows:
        data([row["Symbol"] for row in aggrid.selected_rows])


if __name__ == "__main__":
    set_page_config_once()
    main()

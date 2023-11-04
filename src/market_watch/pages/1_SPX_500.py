import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder

from market_watch.utils import (
    DATA_DIR,
    display_tickers,
    get_spx_hists,
    set_page_config_once,
)


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
    info = pd.read_parquet(DATA_DIR / "spx_info.parquet")
    constituents = constituents.join(
        info[["marketCap", "exchange"]].rename(
            columns={"marketCap": "Market Cap", "exchange": "Exchange"}
        ),
        on="Symbol",
    )
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
        display_tickers([row["Symbol"] for row in aggrid.selected_rows])


if __name__ == "__main__":
    set_page_config_once()
    main()

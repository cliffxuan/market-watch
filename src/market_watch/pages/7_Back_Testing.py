from typing import Literal, NamedTuple

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from market_watch.strategy import multi_run, run
from market_watch.utils import set_page_config_once
from plotly.subplots import make_subplots


class Transaction(NamedTuple):
    date: pd.Timestamp
    type: Literal["Buy", "Sell"]
    price: float
    profit: float
    size: float
    cash: float


@st.cache_data(ttl="1h")
def get_hist(ticker):
    df = yf.Ticker(ticker).history(period="10y")
    df["Date"] = df.index.date  # type: ignore
    return df.set_index(["Date"])


def main():
    st.markdown("# Back Testing Trading Strategy")
    input_cols = st.columns(3)
    ticker = input_cols[0].selectbox(
        "yahoo finance ticker", options=["MSFT", "AAPL", "NVDA", "GOOGL", "BTC-USD"]
    )
    start_capital = input_cols[1].number_input("start capital", value=10_000)
    fee = input_cols[2].number_input("fee %", value=0.2) / 100
    st.markdown("## Moving Average Cross-over")
    ma_col_0, ma_col_1 = st.columns(2)
    fast_ma = int(ma_col_0.number_input("Fast Moving Average", value=5))
    slow_ma = int(ma_col_1.number_input("Slow Moving Average", value=8))
    hist = get_hist(ticker)
    df = hist.loc[:, ["Close", "Volume"]]
    trade_df, end_capital = run(df, fast_ma, slow_ma, start_capital, fee)

    st.markdown("## Trades")
    result_cols = st.columns(2)
    with result_cols[0]:
        st.dataframe(trade_df)

    with result_cols[1]:
        cols = st.columns(2)
        cols[0].metric("end capital", f"{end_capital:,.0f}")
        cols[1].metric("profit", f"{(end_capital / start_capital -1 )* 100:,.2f} %")

        buy_and_hold = (
            trade_df.iloc[0]["Size"] * df.iloc[-1]["Price"] + trade_df.iloc[0]["Cash"]
        )
        cols[0].metric("buy & hold", f"{buy_and_hold:,.0f}")
        cols[1].metric(
            "buy & hold profit", f"{buy_and_hold / start_capital * 100:,.2f} %"
        )

        winning_trades = trade_df[trade_df["Profit"] > 0]
        winning_rate = len(winning_trades) / len(trade_df)
        cols[0].metric("total trades", f"{len(trade_df):,}")
        cols[1].metric("winning rate", f"{winning_rate * 100:,.2f} %")

    buy_df = trade_df[trade_df["TX"] == "Buy"]
    sell_df = trade_df[trade_df["TX"] == "Sell"]
    line = px.line(
        df[["Price", f"{fast_ma}MA", f"{slow_ma}MA"]],  # TODO: dedup
        width=1024,
        height=768,
    )
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
    for trace in line.data:
        fig.append_trace(trace, row=1, col=1)
    fig.append_trace(
        px.scatter(
            buy_df,
            x=buy_df.index,
            y="Price",
            color_discrete_sequence=["green"],
            custom_data=["Profit"],
        )
        .update_traces(
            marker={"symbol": "triangle-up"},
            hovertemplate="<br>".join(
                ["Date: %{x|%Y/%m/%d}", "Price: %{y}", "P/L: %{customdata[0]:.2%}"]
            ),
        )
        .data[0],
        row=2,
        col=1,
    )
    fig.append_trace(
        px.scatter(
            sell_df,
            x=sell_df.index,
            y="Price",
            color_discrete_sequence=["red"],
            custom_data=["Profit"],
        )
        .update_traces(
            marker={"symbol": "triangle-down"},
            hovertemplate="<br>".join(
                ["Date: %{x|%Y/%m/%d}", "Price: %{y}", "P/L: %{customdata[0]:.2%}"]
            ),
        )
        .data[0],
        row=2,
        col=1,
    )
    fig["layout"].update(width=1024, height=768)
    st.plotly_chart(fig, config={"scrollZoom": True}, use_container_width=True)

    refine_input_cols = st.columns(3)
    min_slow_length = int(refine_input_cols[0].number_input("min slow length", value=1))
    max_slow_length = int(
        refine_input_cols[1].number_input("max slow length", value=30)
    )
    interval = int(refine_input_cols[2].number_input("interval", value=1))
    if st.button("refine parameters"):
        results = []
        i = 0
        progress_bar = st.progress(i, text="start")
        total_iteration = sum(
            (slow_ma - 1) // interval
            for slow_ma in range(min_slow_length, max_slow_length + 1, interval)
        )
        increment = 1 / total_iteration
        for fast_ma, slow_ma, end_capital, number_of_trades in multi_run(
            df,
            start_capital,
            fee,
            min_slow_length=min_slow_length,
            max_slow_length=max_slow_length,
            interval=interval,
        ):
            i += increment
            results.append(
                {
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "end_capital": end_capital,
                    "number_of_trades": number_of_trades,
                }
            )
            progress_bar.progress(
                i,
                f"fast: {fast_ma}, slow: {slow_ma}, end capital: {end_capital}",
            )
        if i < 100:
            progress_bar.progress(100, "finish")
        st.dataframe(pd.DataFrame(results))
        st.text(f"expected count: {total_iteration} count: {len(results)}")


if __name__ == "__main__":
    set_page_config_once()
    main()

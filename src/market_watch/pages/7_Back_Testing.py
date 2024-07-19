from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
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
        "yahoo finance ticker", options=["MSFT", "AAPL", "NVDA", "GOOGL"]
    )
    capital = input_cols[1].number_input("starting capital", value=10_000)
    fee = input_cols[2].number_input("fee %", value=0.2) / 100
    st.markdown("## Moving Average Cross-over")
    ma_col_0, ma_col_1 = st.columns(2)
    fast_ma = int(ma_col_0.number_input("Fast Moving Average", value=10))
    slow_ma = int(ma_col_1.number_input("Slow Moving Average", value=20))
    hist = get_hist(ticker)
    df = hist.loc[:, ["Close", "Volume"]]
    fast_ma_name = f"{fast_ma}MA"
    slow_ma_name = f"{slow_ma}MA"
    df[fast_ma_name] = hist["Close"].rolling(window=fast_ma).mean()
    df[slow_ma_name] = hist["Close"].rolling(window=slow_ma).mean()
    df["Price"] = hist["Close"]
    df["diff"] = df[fast_ma_name] - df[slow_ma_name]
    buy_dates = []
    sell_dates = []
    for d_0, d_1, d_2 in zip(df.index, df.index[1:], df.index[2:]):
        if df.loc[d_0]["diff"] < 0 and df.loc[d_1]["diff"] > 0:
            buy_dates.append(d_2)
        elif df.loc[d_0]["diff"] > 0 and df.loc[d_1]["diff"] < 0:
            sell_dates.append(d_2)
    cash = capital
    trades = []
    if buy_dates and sell_dates:
        if buy_dates[0] > sell_dates[0]:
            # TODO how about buying before this?
            sell_dates = sell_dates[1:]
        # TODO how about trailing sell?
        for buy_date, sell_date in zip(buy_dates, sell_dates):
            if buy_date > sell_date:
                raise ValueError(f"cannot buy on {buy_date} after sell on {sell_date}")
            buy_price = df.loc[buy_date]["Price"]
            last_sell_price = trades[-1][2] if trades else buy_price
            buy_profit = (last_sell_price - buy_price) / last_sell_price  # TODO check
            sell_price = df.loc[sell_date]["Price"]
            sell_profit = (sell_price - buy_price) / buy_price
            size = np.floor(cash / (buy_price * (1 + fee)))
            cash -= size * (buy_price * (1 + fee))
            trades.append(
                Transaction(buy_date, "Buy", buy_price, buy_profit, size, cash)
            )
            cash += sell_price * size * (1 - fee)
            trades.append(
                Transaction(sell_date, "Sell", sell_price, sell_profit, -size, cash)
            )
    trade_df = pd.DataFrame(
        trades, columns=["Date", "TX", "Price", "Profit", "Size", "Cash"]
    ).set_index(["Date"])

    st.markdown("## Trades")
    result_cols = st.columns(2)
    with result_cols[0]:
        st.dataframe(trade_df)

    with result_cols[1]:
        last_trade = trade_df.iloc[-1]
        if last_trade["TX"] == "Sell":
            end_value = last_trade["Cash"]
        else:
            end_value = last_trade["Size"] * last_trade["Price"]
        cols = st.columns(2)
        cols[0].metric("final capital", f"{end_value:,.0f}")
        cols[1].metric("profit", f"{end_value / capital * 100:,.2f} %")

        buy_and_hold = (
            trade_df.iloc[0]["Size"] * df.iloc[-1]["Price"] + trade_df.iloc[0]["Cash"]
        )
        cols[0].metric("buy & hold", f"{buy_and_hold:,.0f}")
        cols[1].metric("buy & hold profit", f"{buy_and_hold / capital * 100:,.2f} %")

        winning_trades = trade_df[trade_df["Profit"] > 0]
        winning_rate = len(winning_trades) / len(trades)
        cols[0].metric("total trades", f"{len(trade_df):,}")
        cols[1].metric("winning rate", f"{winning_rate * 100:,.2f} %")

    buy_df = trade_df[trade_df["TX"] == "Buy"]
    sell_df = trade_df[trade_df["TX"] == "Sell"]
    line = px.line(
        df[["Price", fast_ma_name, slow_ma_name]],
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


if __name__ == "__main__":
    set_page_config_once()
    main()

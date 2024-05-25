from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from market_watch.utils import DATA_DIR, set_page_config_once
from plotly.subplots import make_subplots


class Transaction(NamedTuple):
    date: pd.Timestamp
    type: Literal["Buy", "Sell"]
    price: float
    profit: float
    size: float
    cash: float


@st.cache_data
def get_hist(ticker, start=None, end=None):
    df = pd.read_csv(DATA_DIR / "GOOG.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index(["Date"])
    # return yf.Ticker(ticker).history(start=start, end=end)


def main():
    capital = 10_000
    fee = 0.002
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
    st.markdown("# Back Testing")
    hist = get_hist("MSFT", "2004-8-19", "2013-03-01")
    df = hist.loc[:, ["Close", "Volume"]]
    df["10SMA"] = hist["Close"].rolling(window=10).mean()
    df["20SMA"] = hist["Close"].rolling(window=20).mean()
    df["Price"] = hist["Close"]
    df["diff"] = df["10SMA"] - df["20SMA"]
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
            cash -= size * (buy_price + (1 + fee))
            trades.append(
                Transaction(buy_date, "Buy", buy_price, buy_profit, size, cash)
            )
            cash += sell_price * size * (1 - fee)
            trades.append(
                Transaction(sell_date, "Sell", sell_price, sell_profit, -size, cash)
            )
    line = px.line(
        df[["Price", "10SMA", "20SMA"]],
        width=1024,
        height=768,
    )
    trade_df = pd.DataFrame(
        trades, columns=["Date", "TX", "Price", "Profit", "Size", "Cash"]
    ).set_index(["Date"])
    buy_df = trade_df[trade_df["TX"] == "Buy"]
    sell_df = trade_df[trade_df["TX"] == "Sell"]
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
    st.markdown("## Trades")
    st.dataframe(trade_df)


if __name__ == "__main__":
    set_page_config_once()
    main()

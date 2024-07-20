import datetime as dt
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


def run(df, fast_ma, slow_ma, start_capital, fee):
    fast_ma_name = f"{fast_ma}MA"
    slow_ma_name = f"{slow_ma}MA"
    df[fast_ma_name] = df["Close"].rolling(window=fast_ma).mean()
    df[slow_ma_name] = df["Close"].rolling(window=slow_ma).mean()
    df["Price"] = df["Close"]
    df["diff"] = df[fast_ma_name] - df[slow_ma_name]
    buy_dates = []
    sell_dates = []
    order_open = False
    for d_0, d_1, d_2 in zip(df.index, df.index[1:], df.index[2:]):
        if df.loc[d_0]["diff"] < 0 and (
            df.loc[d_1]["diff"] > 0
            or (df.loc[d_1]["diff"] == 0 and df.loc[d_2]["diff"] > 0)
        ):
            if not order_open:
                buy_dates.append(d_2)
                order_open = True
            else:
                st.dataframe(df)
                st.write([d_0, d_1, d_2])
                raise ValueError("cannot open another order")
        elif df.loc[d_0]["diff"] > 0 and (
            df.loc[d_1]["diff"] < 0
            or (df.loc[d_1]["diff"] == 0 and df.loc[d_2]["diff"] < 0)
        ):
            if order_open:
                sell_dates.append(d_2)
                order_open = False
            elif buy_dates:
                st.dataframe(
                    df.loc[d_0 - dt.timedelta(days=20) : d_0 + dt.timedelta(days=20)]
                )
                st.write([d_0, d_1, d_2])
                raise ValueError("no order to close")

    cash = start_capital
    trades = []
    if buy_dates and sell_dates:
        # if buy_dates[0] > sell_dates[0]:
        #     # TODO how about buying before this?
        #     sell_dates = sell_dates[1:]
        # TODO how about trailing sell?
        for buy_date, sell_date in zip(buy_dates, sell_dates):
            if buy_date > sell_date:
                raise ValueError(f"cannot buy on {buy_date} after sell on {sell_date}")
            if trades and buy_date < trades[-1].date:
                raise ValueError(
                    f"buy date {buy_date} before the last trade at {trades[-1].date}"
                )
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
    if trades:
        last_trade = trade_df.iloc[-1]
        if last_trade["TX"] == "Sell":
            end_capital = last_trade["Cash"]
        else:
            end_capital = last_trade["Size"] * last_trade["Price"]
    else:
        end_capital = start_capital
    return trade_df, end_capital


def multi_run(df, start_capital, fee, min_ma=1, max_ma=50, step=1):
    for slow_ma in range(1, max_ma + 1, step):
        for fast_ma in range(1, slow_ma, step):
            if fast_ma < min_ma or slow_ma == fast_ma or slow_ma == 1:
                continue
            try:
                trade_df, end_capital = run(df, fast_ma, slow_ma, start_capital, fee)
            except Exception as exc:
                raise Exception(
                    f"error running fast_ma: {fast_ma} slow_ma: {slow_ma} start_capital: {start_capital} fee: {fee}"
                ) from exc
            yield fast_ma, slow_ma, end_capital, len(trade_df)


def main():
    st.markdown("# Back Testing Trading Strategy")
    input_cols = st.columns(3)
    ticker = input_cols[0].selectbox(
        "yahoo finance ticker", options=["MSFT", "AAPL", "NVDA", "GOOGL", "BTC-USD"]
    )
    start_capital = input_cols[1].number_input("starting capital", value=10_000)
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
        cols[0].metric("final capital", f"{end_capital:,.0f}")
        cols[1].metric("profit", f"{end_capital / start_capital * 100:,.2f} %")

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

    if st.button("discover best parameters"):
        results = []
        i = 0
        progress_bar = st.progress(i, text="start")
        min_ma = 1
        max_ma = 30
        step = 1
        expected_count = int(
            ((max_ma - min_ma) / step) * ((max_ma - min_ma) / step - 1) * 2 / 3
        )
        increment = 1 / expected_count
        for fast_ma, slow_ma, end_capital, number_of_trades in multi_run(
            df, start_capital, fee, min_ma=min_ma, max_ma=max_ma, step=step
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
                min(i, 100),
                f"fast: {fast_ma}, slow: {slow_ma}, end capital: {end_capital}",
            )
        if i < 100:
            progress_bar.progress(100, "finish")
        st.dataframe(pd.DataFrame(results))
        st.text(f"expected count: {expected_count} count: {len(results)}")


if __name__ == "__main__":
    set_page_config_once()
    main()

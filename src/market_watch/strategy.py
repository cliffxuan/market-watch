from typing import Literal, NamedTuple

import numpy as np
import pandas as pd


class Transaction(NamedTuple):
    date: pd.Timestamp
    type: Literal["Buy", "Sell"]
    price: float
    profit: float
    size: float
    cash: float


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
        if (
            df.loc[d_0]["diff"] < 0
            and df.loc[d_1]["diff"] >= 0
            and df.loc[d_2]["diff"] > 0
            and not order_open
        ):
            buy_dates.append(d_2)
            order_open = True
        elif (
            df.loc[d_0]["diff"] > 0
            and df.loc[d_1]["diff"] <= 0
            and df.loc[d_2]["diff"] < 0
            and order_open
        ):
            sell_dates.append(d_2)
            order_open = False

    cash = start_capital
    trades = []
    if buy_dates and sell_dates:
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
            end_capital = int(last_trade["Cash"])
        else:
            end_capital = int(last_trade["Size"] * last_trade["Price"])
    else:
        end_capital = start_capital
    return trade_df, end_capital


def multi_run(
    df, start_capital, fee, min_slow_length=1, max_slow_length=50, interval=1
):
    for slow_ma in range(min_slow_length, max_slow_length + 1, interval):
        for fast_ma in range(1, slow_ma, interval):
            try:
                trade_df, end_capital = run(
                    df.copy(), fast_ma, slow_ma, start_capital, fee
                )
            except Exception as exc:
                raise Exception(
                    f"error running fast_ma: {fast_ma} slow_ma: {slow_ma} start_capital: {start_capital} fee: {fee}"
                ) from exc
            yield fast_ma, slow_ma, end_capital, len(trade_df)

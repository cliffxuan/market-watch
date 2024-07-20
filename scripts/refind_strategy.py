from pathlib import Path

import pandas as pd
import typer
import yfinance as yf
from market_watch.strategy import multi_run

app = typer.Typer()


PWD = Path(__file__).absolute().parent


@app.command()
def two_ma(
    ticker: str = "BTC-USD",
    min_fast: int = 1,
    max_slow: int = 20,
    interval: int = 1,
    period: str = "10y",
    start_capital: int = 10_000,
    fee: float = 0.002,
):
    df = yf.Ticker(ticker).history(period=period)
    total_iteration = 0
    for slow_ma in range(1, max_slow + 1, interval):
        for fast_ma in range(1, slow_ma, interval):
            if fast_ma < min_fast or slow_ma == fast_ma or slow_ma == 1:
                continue
            total_iteration += 1
    inc = 1 / total_iteration
    i = 0
    results = []
    for fast_ma, slow_ma, end_capital, number_of_trades in multi_run(
        df,
        start_capital,
        fee,
        min_fast_length=min_fast,
        max_slow_length=max_slow,
        interval=interval,
    ):
        i += 1
        # results.append(
        #     {
        #         "fast_ma": fast_ma,
        #         "slow_ma": slow_ma,
        #         "end_capital": end_capital,
        #         "number_of_trades": number_of_trades,
        #     }
        # )
        results.append(
            [
                fast_ma,
                slow_ma,
                end_capital,
                number_of_trades,
            ]
        )
        print(
            f"{i * inc * 100:.2f} %   fast: {fast_ma}, slow: {slow_ma}, profit: {end_capital / start_capital * 100:,.1f}%, #trades: {number_of_trades}",
            end="\r",
        )
    output = PWD / f"{ticker}-2ma.csv"
    result_df = pd.DataFrame(
        results,
        columns=[
            "fast_ma",
            "slow_ma",
            "end_capital",
            "number_of_trades",
        ],
    ).sort_values(by="end_capital", ascending=False)
    result_df.to_csv(output)
    print("\nfinished!")
    print(result_df.head(10))
    print(f"saved result to {output}")


if __name__ == "__main__":
    app()

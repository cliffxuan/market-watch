from pathlib import Path

import pandas as pd
import typer
import yfinance as yf
from rich import print

import market_watch.binance as bn
from market_watch.strategy import multi_run

app = typer.Typer()


PWD = Path(__file__).absolute().parent
DATA_DIR = PWD.parent / "data"


@app.command()
def yf_two_ma(
    ticker: str = "BTC-USD",
    min_slow: int = 1,
    max_slow: int = 20,
    step: int = 1,
    period: str = "10y",
    start_capital: int = 10_000,
    fee: float = 0.002,
    out_dir: str = str(DATA_DIR / "strategies" / "yf-two-ma"),
):
    df = yf.Ticker(ticker).history(period=period)
    total_iteration = sum(
        (slow_ma - 1) // step for slow_ma in range(min_slow, max_slow + 1, step)
    )
    inc = 1 / total_iteration
    i = 0
    results = []
    for fast_ma, slow_ma, end_capital, number_of_trades in multi_run(
        df,
        start_capital,
        fee,
        min_slow_length=min_slow,
        max_slow_length=max_slow,
        step=step,
    ):
        i += 1
        profit = end_capital / start_capital - 1
        results.append(
            [
                fast_ma,
                slow_ma,
                profit,
                end_capital,
                number_of_trades,
            ]
        )
        print(
            f"{i * inc * 100:.2f} %  {i} / {total_iteration}"
            f" fast: {fast_ma}, slow: {slow_ma}, profit: {profit * 100:,.1f}%,"
            f" #trades: {number_of_trades}",
            end="\r",
        )
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    output = Path(out_dir) / f"{ticker}-2ma-{min_slow:03}-{max_slow:03}.csv"
    result_df = pd.DataFrame(
        results,
        columns=[
            "fast_ma",
            "slow_ma",
            "profit",
            "end_capital",
            "number_of_trades",
        ],
    ).sort_values(by="end_capital", ascending=False)
    result_df.to_csv(output)
    print("\nfinished!")
    print(result_df.head(10))
    print(f"saved result to {output}")


@app.command()
def bn_two_ma(
    symbol: str = "BTCUSDT",
    min_slow: int = 1,
    max_slow: int = 20,
    step: int = 1,
    interval: str = "4h",
    limit: int = 1000,
    start_capital: int = 10_000,
    fee: float = 0.002,
    ema: bool = False,
    out_dir: str = str(DATA_DIR / "strategies" / "bn-two-ma"),
):
    df = bn.history(symbol, interval, limit)
    total_iteration = sum(
        (slow_ma - 1) // step for slow_ma in range(min_slow, max_slow + 1, step)
    )
    inc = 1 / total_iteration
    i = 0
    results = []
    for fast_ma, slow_ma, end_capital, number_of_trades in multi_run(
        df,
        start_capital,
        fee,
        min_slow_length=min_slow,
        max_slow_length=max_slow,
        step=step,
        fraction=True,
        ema=ema,
    ):
        i += 1
        profit = end_capital / start_capital - 1
        results.append(
            [
                fast_ma,
                slow_ma,
                profit,
                end_capital,
                number_of_trades,
            ]
        )
        print(
            f"{i * inc * 100:.2f} %  {i} / {total_iteration}"
            f" fast: {fast_ma}, slow: {slow_ma}, profit: {profit * 100:,.1f}%,"
            f" #trades: {number_of_trades}",
            end="\r",
        )
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    output = Path(out_dir) / f"{symbol}-2ma-{min_slow:03}-{max_slow:03}.csv"
    result_df = pd.DataFrame(
        results,
        columns=[
            "fast_ma",
            "slow_ma",
            "profit",
            "end_capital",
            "number_of_trades",
        ],
    ).sort_values(by="end_capital", ascending=False)
    result_df.to_csv(output)
    print("\nfinished!")
    print(result_df.head(10))
    print(f"saved result to {output}")


@app.command()
def analyse(data_dir: str, head: int = 100):
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        print(f'[red]data dir "{data_dir_path}" does not exist![/red]')
    dfs = []
    for f in data_dir_path.glob("*.csv"):
        print(f"read file [green]{f}[/green]")
        df = pd.read_csv(f)
        dfs.append(df.drop(df.columns[0], axis=1))
    df = pd.concat(dfs)
    df = df.sort_values(by="profit", ascending=False).reset_index(drop=True)
    pd.set_option("display.max_rows", head)
    print(df.head(head))


if __name__ == "__main__":
    app()

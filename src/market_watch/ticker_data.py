from pathlib import Path

import pandas as pd
import yfinance as yf

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent.parent / "data"


def rank_by_market_cap(constituents: pd.DataFrame) -> pd.DataFrame:
    constituents = constituents.sort_values(
        by=["Market Cap"], ascending=False
    ).reset_index(drop=True)
    goog = constituents.Symbol.loc[lambda x: x.isin(["GOOGL", "GOOG"])]
    if len(goog) > 0:
        rank = constituents.index.map(lambda n: n + 1 if n <= goog.index.min() else n)
    else:
        rank = constituents.index.map(lambda n: n + 1)
    constituents.insert(0, "Rank", rank)
    return constituents


def get_tickers_hists(tickers: list[str]) -> pd.DataFrame:
    unique_columns = [
        "Close",
        "Dividends",
        "High",
        "Low",
        "Open",
        "Stock Splits",
        "Volume",
    ]
    combined_data = {}
    for i, ticker in enumerate(tickers):
        print(
            f"{str(i).zfill(len(str(len(tickers))))} / {len(tickers)} {ticker}",
            end="\r",
        )
        try:
            df = yf.Ticker(ticker).history(period="10y", raise_errors=True)
        except yf.exceptions.YFInvalidPeriodError:  # type: ignore
            df = yf.Ticker(ticker).history(period="max", raise_errors=False)
        for col in unique_columns:
            combined_data[(col, ticker)] = df[col]

    columns = pd.MultiIndex.from_tuples(
        [(col, ticker) for ticker in tickers for col in unique_columns],
        names=["Price", "Ticker"],
    )

    # Create the combined DataFrame
    combined_df = pd.DataFrame(combined_data, columns=columns)
    return combined_df


def calculate_returns(
    data: dict, close_prices: pd.DataFrame, symbols: list[str]
) -> pd.DataFrame:
    # Create DataFrame from tickers info
    constituents = pd.DataFrame.from_dict(
        {
            symbol: {
                "Symbol": symbol,
                "Name": val["price"]["shortName"],
                "Market Cap": val["price"]["marketCap"]["raw"],
                "Volume": val["summaryDetail"]["volume"]["raw"],
                "V/C ‱": (
                    round(
                        val["summaryDetail"]["volume"]["raw"] / marketcap * 10_000,
                        2,
                    )
                    if (marketcap := val["price"]["marketCap"]["raw"]) != 0
                    else None
                ),
            }
            for symbol in symbols
            if (val := data.get(symbol))
        },
        orient="index",
    )

    periods = {
        1: "1d",
        3: "3d",
        7: "7d",
        15: "15d",
        30: "30d",
        90: "90d",
        180: "6mo",
        365: "1y",
        730: "2y",
        1095: "3y",
        1460: "4y",
    }
    for period, label in periods.items():
        constituents[f"{label} %"] = (
            close_prices.iloc[-1] / close_prices.iloc[-period - 1] * 100 - 100
        ).round(2)

    constituents = rank_by_market_cap(constituents)
    return constituents

# pipedream add-package pyarrow
import argparse
import datetime as dt
import gzip
import os
import time
from pathlib import Path

import orjson
import pandas as pd
import yfinance as yf
from github import Auth, Github

from market_watch import yahoo_finance as yf2

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent / "data"
assert DATA_DIR.exists()
HIST_PARQUET = "hist.parquet"
INFO_JSON_GZ = "info.json.gz"


def get_tickers(local: bool = True) -> list[str]:
    if local:
        spx_symbols = pd.read_csv(DATA_DIR / "spx-500.csv")["Symbol"]
        # csv file is from https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index
        nasdqd_symbols = pd.read_csv(DATA_DIR / "nasdaq-100.csv")["Symbol"]
    else:
        spx_symbols = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        )["Symbol"]
        nasdqd_symbols = pd.read_csv(
            "https://raw.githubusercontent.com/cliffxuan/market-watch/main/data/nasdaq-100.csv"
        )["Symbol"]
    symbols = pd.concat([spx_symbols, nasdqd_symbols]).drop_duplicates()
    return symbols.apply(lambda x: x.replace(".", "-")).sort_values().to_list()


def get_hists(local: bool = True) -> pd.DataFrame:
    tickers = get_tickers(local)
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


def get_info(local: bool = True) -> pd.DataFrame:
    info = {}
    tickers = get_tickers(local)
    for i, ticker in enumerate(tickers):
        print(
            f"{str(i).zfill(len(str(len(tickers))))} / {len(tickers)} {ticker}",
            end="\r",
        )
        info[ticker] = yf.Ticker(ticker).info  # yf.Ticker.info is broken
    return pd.DataFrame.from_dict(info, orient="index").sort_values(
        by="marketCap", ascending=False
    )


def get_info_json(local: bool = True) -> dict[str, dict]:
    info = {}
    failed = []
    tickers = get_tickers(local)
    for i, ticker in enumerate(tickers):
        for retry in range(1, 5):
            print(
                f"{str(i).zfill(len(str(len(tickers))))} / {len(tickers)} {ticker}",
                end="\r",
            )
            try:
                info[ticker] = yf2.get_info(ticker)
                time.sleep(0.1 * retry)
                break
            except Exception:
                pass
        else:
            failed.append(ticker)
    if failed:
        print(f"{len(failed)} tickes failed:", failed)
    return dict(
        sorted(
            info.items(),
            key=lambda item: item[1]["summaryDetail"]["marketCap"]["raw"],
            reverse=True,
        )
    )


def commit(file_path: Path, message: str = "") -> None:
    with Github(auth=Auth.Token(os.environ["GITHUB_AUTH_TOKEN"])) as g:
        repo = g.get_repo("cliffxuan/market-watch")
        contents = repo.get_contents(f"data/{HIST_PARQUET}", ref="main")
        new_content = open(file_path, "rb").read()
        print("push to github")
        repo.update_file(
            path=contents.path,
            message=message
            or f"updated {HIST_PARQUET} @ {dt.datetime.now().isoformat()}",
            content=new_content,
            sha=contents.sha,
            branch="main",
        )
    print("succeed!")


def handler(pd: "pipedream"):  # type: ignore  # noqa
    """
    pipedream handler
    """
    print(pd.steps["trigger"]["context"]["id"])
    df = get_hists(local=False)
    file_path = Path("/tmp") / HIST_PARQUET
    print(f"write file to {file_path}")
    df.to_parquet(file_path, compression="gzip")
    message = f"updated {HIST_PARQUET} @ {dt.datetime.now().isoformat()} from pipedream"
    commit(file_path, message=message)
    return {"succeed": True, "message": message}


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="etl")
    parser.add_argument("--commit", "-c", action="store_true", help="commit to github")
    parser.add_argument("--all", "-a", action="store_true", help="get all")
    parser.add_argument("--price", "-p", action="store_true", help="get price")
    parser.add_argument("--info", "-i", action="store_true", help="get info")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = argument_parser().parse_args(argv)
    if args.all or args.price:
        hist_file_path = DATA_DIR / HIST_PARQUET
        print(f"get hist data and write to {hist_file_path}")
        get_hists().to_parquet(hist_file_path, compression="gzip")
        if args.commit:
            commit(hist_file_path)

    if args.all or args.info:
        info_file_path = DATA_DIR / INFO_JSON_GZ
        print(f"get info data and write to {info_file_path}")
        info = get_info_json()
        with open(info_file_path, "wb") as f:
            f.write(gzip.compress(orjson.dumps(info, option=orjson.OPT_INDENT_2)))
        if args.commit:
            commit(info_file_path)


if __name__ == "__main__":
    print(get_tickers())
    main()

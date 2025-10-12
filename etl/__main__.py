# pipedream add-package pyarrow
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
from market_watch.ticker_data import calculate_returns, get_tickers_hists

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent / "data"
assert DATA_DIR.exists()
HIST_PARQUET = "hist.parquet"
INFO_JSON_GZ = "info.json.gz"

COLLECTIONS = [
    "spx-500",
    "nasdaq-100",
    "ark-innovation",
]


def get_tickers(local: bool = True) -> list[str]:
    if local:
        dfs = [
            pd.read_csv(DATA_DIR / "raw" / f"{collection}.csv")["Symbol"]
            for collection in COLLECTIONS
        ]
    else:
        dfs = [
            pd.read_csv(
                f"https://raw.githubusercontent.com/cliffxuan/market-watch/main/data/raw/{collection}.csv"
            )["Symbol"]
            for collection in COLLECTIONS
        ]
    symbols = pd.concat(dfs).drop_duplicates().dropna()
    return symbols.apply(lambda x: x.replace(".", "-")).sort_values().to_list()


def get_hists(local: bool = True) -> pd.DataFrame:
    return get_tickers_hists(get_tickers(local))


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


def main() -> None:
    hist_file_path = DATA_DIR / HIST_PARQUET
    info_file_path = DATA_DIR / INFO_JSON_GZ

    print(f"get hist data and write to {hist_file_path}")
    hists = get_hists()
    hists.to_parquet(hist_file_path, compression="gzip")

    print(f"get info data and write to {info_file_path}")
    info = get_info_json()
    with open(info_file_path, "wb") as f:
        f.write(gzip.compress(orjson.dumps(info, option=orjson.OPT_INDENT_2)))

    for collection in COLLECTIONS:
        df = calculate_returns(
            data=info,
            close_prices=pd.read_parquet(hist_file_path)["Close"],
            symbols=pd.read_csv(DATA_DIR / "raw" / f"{collection}.csv")[
                "Symbol"
            ].tolist(),
        )
        for file in (
            "latest.csv",
            f"{dt.datetime.now().strftime('%Y-%m-%d')}.csv",
        ):
            directory = DATA_DIR / collection
            directory.mkdir(exist_ok=True)
            path = directory / file
            print(f"write to {path}")
            df.to_csv(path, index=False)


if __name__ == "__main__":
    main()

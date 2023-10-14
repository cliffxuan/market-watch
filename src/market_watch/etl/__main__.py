import datetime as dt
import os

import pandas as pd
import yfinance as yf
from github import Auth, Github

from market_watch.utils import DATA_DIR

SPX_HIST = "spx_hist.parquet"


def get_hists() -> pd.DataFrame:
    constituents = pd.read_csv(DATA_DIR / "spx_constituents.csv")
    tickers = constituents["Symbol"].sort_values().to_list()
    return yf.Tickers(tickers).history()


def commit():
    with Github(auth=Auth.Token(os.environ["GITHUB_AUTH_TOKEN"])) as g:
        repo = g.get_repo("cliffxuan/market-watch")
        contents = repo.get_contents(f"data/{SPX_HIST}", ref="main")
        new_content = open(DATA_DIR / SPX_HIST, "rb").read()
        repo.update_file(
            path=contents.path,
            message=f"updated {SPX_HIST} @ {dt.datetime.now().isoformat()}",
            content=new_content,
            sha=contents.sha,
            branch="main",
        )


def main():
    df = get_hists()
    df.to_parquet(DATA_DIR / SPX_HIST, compression="gzip")
    commit()


if __name__ == "__main__":
    main()

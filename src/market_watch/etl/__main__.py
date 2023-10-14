# pipedream add-package pyarrow
import datetime as dt
import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from github import Auth, Github

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent.parent.parent / "data"
SPX_HIST = "spx_hist.parquet"


def get_hists(local: bool = True) -> pd.DataFrame:
    if local:
        constituents = pd.read_csv(DATA_DIR / "spx_constituents.csv")
    else:
        constituents = pd.read_csv(
            "https://raw.githubusercontent.com/cliffxuan/market-watch/main/data/spx_constituents.csv"
        )
    tickers = constituents["Symbol"].sort_values().to_list()
    return yf.Tickers(tickers).history()


def commit(file_path: Path, message: str = ""):
    with Github(auth=Auth.Token(os.environ["GITHUB_AUTH_TOKEN"])) as g:
        repo = g.get_repo("cliffxuan/market-watch")
        contents = repo.get_contents(f"data/{SPX_HIST}", ref="main")
        new_content = open(file_path, "rb").read()
        print("push to github")
        repo.update_file(
            path=contents.path,
            message=message or f"updated {SPX_HIST} @ {dt.datetime.now().isoformat()}",
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
    file_path = Path("/tmp") / SPX_HIST
    print(f"write file to {file_path}")
    df.to_parquet(file_path, compression="gzip")
    message = f"updated {SPX_HIST} @ {dt.datetime.now().isoformat()} from pipedream"
    commit(file_path, message=message)
    return {"succeed": True, "message": message}


def main():
    df = get_hists()
    file_path = DATA_DIR / SPX_HIST
    print(f"write file to {file_path}")
    df.to_parquet(file_path, compression="gzip")
    commit(file_path)


if __name__ == "__main__":
    main()

import datetime as dt
import gzip
from pathlib import Path

import orjson
import pandas as pd

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent.parent / "data"


def rank_by_market_cap(constituents: pd.DataFrame) -> pd.DataFrame:
    constituents = constituents.sort_values(
        by=["Market Cap"], ascending=False
    ).reset_index(drop=True)
    goog = constituents.Symbol.loc[lambda x: x.isin(["GOOGL", "GOOG"])]
    rank = constituents.index.map(lambda n: n + 1 if n <= goog.index.min() else n)
    constituents.insert(0, "Rank", rank)
    return constituents


def get_tickers_info() -> dict:
    file_path = DATA_DIR / "info.json.gz"

    try:
        with open(f"{file_path}.timestamp", "r") as f:
            creation_time = dt.datetime.fromisoformat(f.read())
    except Exception:
        creation_time = None
    with open(file_path, "rb") as f:
        data = orjson.loads(gzip.decompress(f.read()))
    return {"data": data, "creation_time": creation_time}

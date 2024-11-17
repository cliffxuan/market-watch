import pandas as pd

from market_watch.index_table import index_table
from market_watch.utils import DATA_DIR, set_page_config_once

if __name__ == "__main__":
    set_page_config_once()
    constituents = pd.read_csv(DATA_DIR / "nasdaq-100.csv")[["Symbol"]]
    index_table("NASDAQ 100", constituents)

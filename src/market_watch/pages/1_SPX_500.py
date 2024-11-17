import pandas as pd

from market_watch.index_table import index_table
from market_watch.utils import DATA_DIR, set_page_config_once

if __name__ == "__main__":
    set_page_config_once()
    symbols = pd.read_csv(DATA_DIR / "spx-500.csv")["Symbol"].tolist()
    index_table("S&P 500", symbols)

import pandas as pd

from market_watch.index_table import index_table
from market_watch.utils import DATA_DIR, set_page_config_once

if __name__ == "__main__":
    set_page_config_once()
    symbols = pd.read_csv(DATA_DIR / "raw" / "ark-innovation.csv")["Symbol"].tolist()
    index_table("ARK Innovation ETF", symbols)

from market_watch.index_table import index_table
from market_watch.utils import set_page_config_once

if __name__ == "__main__":
    set_page_config_once()
    index_table("NASDAQ 100", "nasdaq-100.csv", ["Symbol", "Name"])

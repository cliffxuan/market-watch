from market_watch.index_table import tickers_table
from market_watch.utils import set_page_config_once

symbols = [
    "NVDA",
    "MSTR",
    "COIN",
    "TSLA",
    "RBLX",
    "SNOW",
    "MARA",
    "CLSK",
]


if __name__ == "__main__":
    set_page_config_once()
    info = {}
    tickers_table("My Watchlist", symbols)

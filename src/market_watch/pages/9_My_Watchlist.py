from market_watch import yahoo_finance as yf2
from market_watch.index_table import index_table
from market_watch.ticker_data import get_tickers_hists
from market_watch.utils import set_page_config_once

if __name__ == "__main__":
    set_page_config_once()
    symbols = ["MSTR", "COIN", "TSLA"]
    info = {}
    index_table(
        "My Watchlist",
        symbols,
        tickers_info={symbol: yf2.get_info(symbol) for symbol in symbols},
        close_prices=get_tickers_hists(symbols)["Close"],
    )

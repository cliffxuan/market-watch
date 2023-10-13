import pandas as pd
import yfinance as yf

from market_watch.utils import DATA_DIR


def get_hists() -> pd.DataFrame:
    constituents = pd.read_csv(DATA_DIR / "spx_constituents.csv")
    tickers = constituents["Symbol"].sort_values().to_list()
    return yf.Tickers(tickers).history()


def main():
    df = get_hists()
    df.to_parquet(DATA_DIR / "spx_hist.parquet", compression="gzip")


if __name__ == "__main__":
    main()

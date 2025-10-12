"""
get the list from here:

nasdaq-100:
https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index

spx-500:
https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from curl_cffi import requests

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent / "data"

TYPES = ["spx-500", "nasdaq-100"]


def get_spx_500() -> pd.DataFrame:
    print("Fetching SPX-500 constituents...")
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    constituents = pd.read_csv(url)
    print(f"Retrieved {len(constituents)} constituents from {url}")
    print("\nTop 5 constituents:")
    print(constituents.head())
    return constituents


def get_nasdaq_100() -> pd.DataFrame:
    print("Fetching NASDAQ-100 constituents...")
    api_url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
    response = requests.get(api_url, impersonate="chrome")
    response.raise_for_status()
    data = response.json()
    constituents = pd.DataFrame(data["data"]["data"]["rows"])
    # Convert marketCap to integer
    constituents["marketCap"] = (
        constituents["marketCap"].replace({r"\$": "", ",": ""}, regex=True).astype(int)
    )
    # Sort by marketCap in descending order
    constituents = constituents.sort_values(by="marketCap", ascending=False)
    print(f"Retrieved {len(constituents)} constituents from {api_url}")
    print("\nTop 5 constituents:")
    print(constituents.head())
    return constituents


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="process nasdaq 100")
    parser.add_argument("--type", "-t", choices=TYPES, help="name of the list type")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = argument_parser().parse_args(argv)
    if args.type in ("spx-500", None):
        companies = get_spx_500()
        companies["Symbol"] = companies["Symbol"].apply(lambda x: x.replace(".", "-"))
        companies.to_csv(DATA_DIR / "raw" / "spx-500.csv", index=False)
    if args.type in ("nasdaq-100", None):
        companies = get_nasdaq_100()
        companies.columns = [col.capitalize() for col in companies.columns]
        companies.to_csv(DATA_DIR / "raw" / "nasdaq-100.csv", index=False)


if __name__ == "__main__":
    main()

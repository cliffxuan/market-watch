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

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent / "data"

TYPES = ["spx-500", "nasdaq-100"]


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="process nasdaq 100")
    parser.add_argument(
        "--file",
        "-f",
        type=argparse.FileType("r"),
        help="name of the file to convert",
    )
    parser.add_argument("--type", "-t", choices=TYPES, help="name of the list type")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = argument_parser().parse_args(argv)
    if args.type == "spx-500":
        companies = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        )
        companies["Symbol"] = companies["Symbol"].apply(lambda x: x.replace(".", "-"))
    elif args.type == "nasdaq-100":
        if not args.file:
            raise ValueError(
                "need a csv file. copy and paste from"
                " https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index"
            )
        companies = pd.read_csv(args.file, sep="\t")[["Symbol", "Name"]]
    else:
        raise Exception(f"unsupported list type {args.type}")
    companies.to_csv(DATA_DIR / "raw" / f"{args.type}.csv", index=False)


if __name__ == "__main__":
    main()

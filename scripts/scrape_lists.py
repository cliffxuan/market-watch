import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup

LISTS = {
    "spx-500": "sp500",
    "nasdaq-100": "nasdaq100",
    "ark-innovation": "etf/ark-invest/ARKK",
}
PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent / "data"


def fetch_constituents(name: str) -> None:
    url = f"https://www.slickcharts.com/{LISTS[name]}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", attrs={"class": "table"})
    if not table:
        raise RuntimeError("Could not find the constituents table on the page")

    # Extract headers
    headers_row = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    if headers_row[-1] == "":  # remove the arrow at the end if exists
        headers_row = headers_row[:-1]

    # Extract table data
    data = []
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        row_data = [cell.get_text(strip=True) for cell in cells]
        if row_data[-1] == "":
            row_data = row_data[:-1]
        if row_data:
            data.append(row_data)

    filename = DATA_DIR / "raw" / f"{name}.csv"
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers_row)
        writer.writerows(data)
    print("✅ Successfully fetched:", len(data), "companies")
    print("Example:", data[0])
    print(f"💾 Saved to {filename}")


if __name__ == "__main__":
    for name in LISTS:
        fetch_constituents(name)

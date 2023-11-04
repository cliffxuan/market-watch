from functools import lru_cache
from urllib.parse import quote_plus

import requests

Cookies = tuple[tuple[str, str | None], ...]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/58.0.3029.110 Safari/537.36"
}


@lru_cache
def get_cookies() -> Cookies:
    response = requests.get("https://fc.yahoo.com", headers=HEADERS)
    if not response.cookies:
        raise Exception("Failed to obtain Yahoo auth cookie.")
    return tuple(response.cookies.items())


@lru_cache
def get_crumb(cookies: Cookies) -> str:
    response = requests.get(
        "https://query1.finance.yahoo.com/v1/test/getcrumb",
        headers=HEADERS,
        cookies=dict(cookies),
    )
    response.raise_for_status()
    crumb = response.text
    if crumb is None:
        raise ValueError("Failed to retrieve Yahoo crumb.")
    return crumb


@lru_cache
def get_info(ticker: str, modules: list[str] | None = None) -> dict:
    default_modules = (
        "assetProfile",
        "secFilings",
        "summaryDetail",
        "financialData",
        "indexTrend",
        "quoteType",
        "price",
        "defaultKeyStatistics",
    )
    cookies = get_cookies()
    crumb = get_crumb(cookies)
    url = (
        "https://query1.finance.yahoo.com/v10/finance/quoteSummary/"
        f"{ticker}?formatted=true&crumb={crumb}&lang=en-US"
        f"&modules={quote_plus(','.join(modules or default_modules))}"
    )
    response = requests.get(url, headers=HEADERS, cookies=dict(cookies))
    response.raise_for_status()
    return response.json()["quoteSummary"]["result"][0]

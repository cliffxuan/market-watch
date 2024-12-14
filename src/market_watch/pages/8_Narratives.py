import logging

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from market_watch.index_table import search
from market_watch.utils import set_page_config_once, trading_view

NARRATIVES = {
    "BRC20": [
        {
            "name": "Stacks",
            "ticker": "STX4847-USD",
            "symbol": "STXUSD",
        },
    ],
    "SOL": [
        {
            "name": "PYTH",
            "ticker": "PYTH-USD",
            "symbol": "PYTHUSD",
        },
        {
            "name": "SOL",
            "ticker": "SOL-USD",
            "symbol": "SOLUSD",
        },
    ],
    "LAYER1": [
        {
            "name": "SUI",
            "ticker": "SUI20947-USD",
            "symbol": "SUIUSD",
        },
        {
            "name": "NEAR Protocol",
            "ticker": "NEAR-USD",
            "symbol": "NEARUSD",
        },
        {
            "name": "Aptos",
            "ticker": "APT21794-USD",
            "symbol": "APTUSD",
        },
    ],
    "AI": [
        {
            "name": "RNDR",
            "ticker": "RNDR-USD",
            "symbol": "RNDRUSD",
            "exchange": "COINBASE",
        },
        {
            "name": "FET",
            "ticker": "FET-USD",
            "symbol": "FETUSD",
        },
        {
            "name": "ai16z",
            "ticker": "AI16Z-USD",
            "symbol": "AI16ZUSDT",
            "exchange": "GATEIO",
        },
        {
            "name": "Virtuals Protocol",
            "ticker": "VIRTUAL-USD",
            "symbol": "VIRTUALUSDT",
            "exchange": "CRYPTO",
        },
        {
            "name": "Goatseus",
            "ticker": "GOAT33440-USD",
            "symbol": "GOATUSDT",
            "exchange": "BYBIT",
        },
    ],
    "Modular": [
        {
            "name": "Celestia",
            "ticker": "TIA22861-USD",
            "symbol": "TIAUSD",
        },
    ],
    "RWA": [
        {
            "name": "ONDO",
            "ticker": "ONDO-USD",
            "symbol": "ONDOUSD",
            "exchange": "COINBASE",
        },
        {
            "name": "CFG",
            "ticker": "CFG-USD",
            "symbol": "CFGUSD",
            "exchange": "KRAKEN",
        },
        {
            "name": "Pendle",
            "ticker": "Pendle-USD",
            "symbol": "PENDLEUSD",
        },
        {
            "name": "GFI",
            "ticker": "GFI13967-USD",
            "symbol": "GFIUSD",
            "exchange": "COINBASE",
        },
        {
            "name": "OM",
            "ticker": "OM-USD",
            "symbol": "OMUSD",
        },
        {
            "name": "AXL",
            "ticker": "AXL17799-USD",
            "symbol": "AXLUSD",
            "exchange": "COINBASE",
        },
    ],
    "BASE": [
        {
            "name": "Aerodrome Finance",
            "ticker": "AERO29270-USD",
            "symbol": "AEROUSD",
            "exchange": "COINBASE",
        }
    ],
    "DePIN": [
        {
            "name": "AIOZ",
            "ticker": "AIOZ-USD",
            "symbol": "AIOZUSD",
            "exchange": "COINBASE",
        },
        {
            "name": "Helium",
            "ticker": "HNT-USD",
            "symbol": "HNTUSD",
            "exchange": "COINBASE",
        },
    ],
    "Gaming": [
        {
            "name": "SuperVerse",
            "ticker": "SUPER8290-USD",
            "symbol": "SUPERUSD",
            "exchange": "BINANCE",
        },
        {
            "name": "Wilder World",
            "ticker": "WILD-USD",
            "symbol": "WILDUSD",
            "exchange": "CRYPTO",
        },
        {
            "name": "Prime",
            "ticker": "WILD-USD",
            "symbol": "PRIMEUSD",
            "exchange": "COINBASE",
        },
        {
            "name": "Beam",
            "ticker": "BEAM28298-USD",
            "symbol": "BEAMXUSDT",
        },
    ],
}

NAME_TO_DATA = {coin["name"]: coin for coins in NARRATIVES.values() for coin in coins}


@st.cache_data(ttl="1h")
def get_hist(name: str) -> pd.DataFrame:
    ticker = yf.ticker.Ticker(name)
    return ticker.history(period="max")[["Open", "High", "Low", "Close", "Volume"]]  # type: ignore


@st.cache_data(ttl="1h")
def get_coin_price(coin: dict) -> dict:
    hist = get_hist(coin["ticker"])
    ticker = yf.ticker.Ticker(coin["ticker"])
    close = hist["Close"]
    try:
        price = close.iloc[-1]
        ath_date = close.idxmax().strftime("%Y-%m-%d")  # type: ignore
        ath_pct = (close.iloc[-1] / close.max() - 1).round(4) * 100
        volume = (hist["Volume"].iloc[-1] / 1_000_000).round(1)
        market_cap = round(ticker.info["marketCap"] / 1_000_000, 1)
        vol_cap_ratio = round(volume / market_cap * 100, 2)
    except Exception:
        logging.exception(f"Error getting price info for coin {coin}")
        price = None
        ath_date = None
        ath_pct = None
        volume = None
        market_cap = None
        vol_cap_ratio = None
    record = {
        "Name": coin["name"],
        "Market Cap": market_cap,
        "Price": price,
        "ATH Date": ath_date,
        "ATH %": ath_pct,
        "Volume": volume,
        "V/C %": vol_cap_ratio,
    }
    for day in [1, 7, 30, 90, 180, 360]:
        try:
            record[f"{day}d %"] = (close.iloc[-1] / close.iloc[-day - 1] - 1).round(
                4
            ) * 100
        except IndexError:
            record[f"{day}d %"] = None
    record["Historical Prices"] = close.values
    return record


@st.cache_data(ttl="1h")
def get_df(narratives: dict[str, list[dict]]) -> pd.DataFrame:
    data = []
    for narrative, coins in narratives.items():
        for coin in coins:
            try:
                record = {"Narrative": narrative, **get_coin_price(coin)}
                data.append(record)
            except Exception:
                logging.exception(f"Error getting {coin}")
    return pd.DataFrame(data)


def display_coins(names: list[str]) -> None:
    tabs = st.tabs(names)
    spike_setting = dict(  # noqa: C408
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        spikecolor="grey",
    )
    for i, name in enumerate(names):
        data = NAME_TO_DATA[name]
        with tabs[i]:
            st.markdown(f"### {data['name']}")
            trading_view(name=data["symbol"], exchange=data.get("exchange", "binance"))
            hist = get_hist(data["ticker"])
            fig = px.line(hist, x=hist.index, y="Close")
            fig.update_xaxes(**spike_setting)
            fig.update_yaxes(**spike_setting)
            fig.update_layout(hoverdistance=0)
            st.plotly_chart(fig)
            st.dataframe(hist)


def main() -> None:
    st.markdown("# Naratives")
    df = get_df(NARRATIVES)
    cols = list(df)
    df["Select"] = st.session_state.setdefault(
        f"{__name__}.selected.{len(df)}", [False] * len(df)
    )
    query = st.text_input(
        "search",
        label_visibility="collapsed",
        placeholder="search",
    )
    df = search(df, query)
    selected = st.data_editor(
        df,
        column_order=["Select", *cols],
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "",
                help="Select to see more info and portfolio optimization",
                default=False,
            ),
            "Historical Prices": st.column_config.LineChartColumn(
                "Historical Prices",
                width="medium",
            ),
            "Market Cap": st.column_config.NumberColumn("Market Cap ($M)"),
            "Volume": st.column_config.NumberColumn("Vol ($M)"),
        },
        disabled=cols,
        hide_index=True,
        height=(len(df) + 1) * 35 + 3,
    )
    names = list(selected[selected["Select"]]["Name"])
    if names:
        display_coins(names)


if __name__ == "__main__":
    set_page_config_once()
    main()

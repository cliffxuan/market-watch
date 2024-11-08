import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

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
}

NAME_TO_DATA = {coin["name"]: coin for coins in NARRATIVES.values() for coin in coins}


@st.cache_data(ttl="1h")
def get_hist(name: str) -> pd.DataFrame:
    ticker = yf.ticker.Ticker(name)
    return ticker.history(period="max")[["Open", "High", "Low", "Close", "Volume"]]  # type: ignore


@st.cache_data(ttl="1h")
def get_df() -> pd.DataFrame:
    data = []
    for narrative in NARRATIVES:
        for coin in NARRATIVES[narrative]:
            hist = get_hist(coin["ticker"])
            ticker = yf.ticker.Ticker(coin["ticker"])
            close = hist["Close"]
            record = {
                "Name": coin["name"],
                "Narrative": narrative,
                "Market Cap": ticker.info["marketCap"],
                "Price": close.iloc[-1],
                "ATH Date": close.idxmax().strftime("%Y-%m-%d"),  # type: ignore
                "ATH %": (close.iloc[-1] / close.max() - 1).round(4) * 100,
            }
            for day in [1, 7, 30, 90, 180, 360]:
                try:
                    record[f"{day}d %"] = (
                        close.iloc[-1] / close.iloc[-day - 1] - 1
                    ).round(4) * 100
                except IndexError:
                    record[f"{day}d %"] = None
            record["Historical Prices"] = close.values
            data.append(record)
    return pd.DataFrame(data)


def display_coins(names: list[str]) -> None:
    tabs = st.tabs(names)
    for i, name in enumerate(names):
        data = NAME_TO_DATA[name]
        with tabs[i]:
            trading_view(name=data["symbol"], exchange=data.get("exchange", "binance"))


def main() -> None:
    st.markdown("# Naratives")

    spike_setting = dict(  # noqa: C408
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        spikecolor="grey",
    )
    debug = True
    debug = False
    df = get_df()
    for narrative in NARRATIVES:
        for coin in NARRATIVES[narrative]:
            hist = get_hist(coin["ticker"])
            if debug:
                st.markdown(f"### {coin['name']}")
                st.dataframe(hist)
                fig = px.line(hist, x=hist.index, y="Close")
                fig.update_xaxes(**spike_setting)
                fig.update_yaxes(**spike_setting)
                fig.update_layout(hoverdistance=0)
                st.plotly_chart(fig)
    cols = list(df)
    df["Select"] = st.session_state.setdefault("{__name__}.selected", [False] * len(df))
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

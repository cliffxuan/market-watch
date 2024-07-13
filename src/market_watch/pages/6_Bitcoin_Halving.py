import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from market_watch.utils import set_page_config_once
from plotly.subplots import make_subplots

PREFIX = "PI_CYCLE"

HALVING_DATES = [
    dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
    for date in [
        "2012-11-28",
        "2016-07-09",
        "2020-05-11",
        "2024-04-20",
    ]
]


@st.cache_data(ttl="1h")
def btc_hist():
    return yf.Ticker("BTC-USD").history(start="2013-04-28")


def main():
    st.markdown("# Bitcoin Halving")
    hist = btc_hist()
    df = hist.loc[:, ["Close", "Volume"]]

    dfs_price_around = {}
    fig = px.line(
        df[["Close"]],
        log_y=True,
        width=1024,
        height=768,
    )
    padding = st.session_state.setdefault(f"{PREFIX}#padding", 5)
    for date in HALVING_DATES:
        fig.add_vline(
            x=date.timestamp() * 1000,
            line_dash="dash",
            line_color="red",
            annotation_text=date.strftime("%Y-%m-%d"),
        )
        dfs_price_around[date] = df[
            (df.index <= date + pd.Timedelta(days=padding))
            & (df.index >= date - pd.Timedelta(days=padding))
        ]
    st.plotly_chart(fig, config={"scrollZoom": True})

    st.markdown("## Local Price Actions")
    st.columns(2)[0].slider(
        "Padding",
        min_value=1,
        max_value=100,
        value=padding,
        key=f"{PREFIX}#padding_slider",
        on_change=lambda: st.session_state.update(
            {f"{PREFIX}#padding": st.session_state[f"{PREFIX}#padding_slider"]}
        ),
    )
    for date, df in dfs_price_around.items():
        if df.empty:
            continue
        st.markdown(f"### {date.strftime('%Y-%m-%d')}")
        st.dataframe(
            df.style.apply(
                lambda row, date: [
                    "background-color: yellow" if row.name == date else ""
                ]
                * len(row),
                date=date,
                axis=1,
            )
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(px.line(df[["Close"]]).data[0])
        fig.add_trace(px.bar(df[["Volume"]], opacity=0.5).data[0], secondary_y=True)
        fig.add_vline(
            x=date.timestamp() * 1000,
            line_dash="dash",
            line_color="red",
            annotation_text=date.strftime("%Y-%m-%d"),
        )
        fig.layout.yaxis2.showgrid = False
        st.plotly_chart(fig)


if __name__ == "__main__":
    set_page_config_once()
    main()

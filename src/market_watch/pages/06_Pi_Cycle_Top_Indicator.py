import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from market_watch.utils import set_page_config_once

PREFIX = "PI_CYCLE"


@st.cache_data(ttl="1h")
def get_hist(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period="max")


def main() -> None:
    st.markdown("# Pi Cycle Top Indicator")
    ticker = st.text_input("ticker", value="BTC-USD")
    hist = get_hist(ticker)
    df = hist.loc[:, ["Close", "Volume"]]
    df["111DMA"] = hist["Close"].rolling(window=111).mean()
    df["350DMA x 2"] = hist["Close"].rolling(window=350).mean() * 2
    df["ratio"] = df["111DMA"] / df["350DMA x 2"]

    df_above = df[df["111DMA"] > df["350DMA x 2"]]
    cross_dates = []
    try:
        prev_date = df_above.index[0]
        cross_dates.append(prev_date)
    except IndexError:
        prev_date = None
    for date in df_above.index:
        if (
            prev_date is not None
            and df.index.get_loc(date) - df.index.get_loc(prev_date) > 1
        ):
            cross_dates.append(date)
        prev_date = date

    dfs_price_around = {}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig = px.line(
        df[["Close", "111DMA", "350DMA x 2"]],
        log_y=True,
        width=1024,
        height=768,
    )
    padding = st.session_state.setdefault(f"{PREFIX}#padding", 5)
    for date in cross_dates:
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
    st.plotly_chart(px.line(df["ratio"]), config={"scrollZoom": True})

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

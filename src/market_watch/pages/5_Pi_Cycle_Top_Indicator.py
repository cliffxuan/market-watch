import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from market_watch.utils import set_page_config_once
from plotly.subplots import make_subplots

PREFIX = "PI_CYCLE"


@st.cache_data(ttl="1h")
def btc_hist():
    return yf.Ticker("BTC-USD").history(start="2013-04-28")


def main():
    st.markdown("# Pi Cycle Top Indicator")
    hist = btc_hist()
    df = hist.loc[:, ["Close", "Volume"]]
    df["111DMA"] = hist["Close"].rolling(window=111).mean()
    df["350DMA x 2"] = hist["Close"].rolling(window=350).mean() * 2

    df_above = df[df["111DMA"] > df["350DMA x 2"]]
    prev_date = df_above.index[0]
    cross_dates = [prev_date]
    for date in df_above.index:
        if prev_date is not None and (date - prev_date).days > 1:
            cross_dates.append(date)
        prev_date = date

    dfs_price_around = {}
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

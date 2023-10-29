import streamlit as st

from js_eval import get_user_agent, is_mobile
from market_watch.utils import display_tickers, set_page_config_once, trading_view


def main():
    st.markdown("# Market Watch")
    if is_mobile():
        st.markdown("Only support desktop")
        st.write(get_user_agent())
        return
    chart_cols = st.columns(2)
    with chart_cols[0]:
        interval = st.radio(
            "Interval",
            ["1H", "4H", "1D", "1W", "1M", "3M", "6M", "12M"],
            horizontal=True,
            index=3,
        )
        col_cnt = int(st.radio("Columns", [1, 2], index=1, horizontal=True))  # type: ignore
    styles = {
        "Bars": 0,
        "Candles": 1,
        "Line": 2,
        "Heikin Ashi": 8,
        "Renko": 4,
        "Point and Figure": 6,
    }
    with chart_cols[1]:
        style = styles[
            st.radio(
                "Bar's Style",
                styles,
                index=1,
                horizontal=True,
            )
        ]  # type: ignore
    chart_cols = st.columns(col_cnt)
    kwargs = {
        "height": 800 // col_cnt,
        "interval": interval,
        "hide_side_toolbar": True,
        "style": style,
    }
    with chart_cols[0]:
        trading_view(
            "SPX500USD", "OANDA", **kwargs
        )  # SP:SPX only avail from tradingview
        trading_view("GOLD", "TVC", **kwargs)
    with chart_cols[-1]:
        trading_view("DXY", "INDEX", **kwargs)
        trading_view("BLX", "BNC", **kwargs)

    display_tickers(["SPX", "DXY", "GOLD", "BTC"])


if __name__ == "__main__":
    set_page_config_once()
    main()

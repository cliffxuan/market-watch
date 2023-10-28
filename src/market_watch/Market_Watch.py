import streamlit as st

from market_watch.utils import set_page_config_once, trading_view


def main():
    st.markdown("# Market Watch")
    interval = st.radio(
        "Interval",
        ["1H", "4H", "1D", "1W", "1M", "3M", "6M", "12M"],
        horizontal=True,
        index=3,
    )
    styles = {
        "Bars": 0,
        "Candles": 1,
        "Line": 2,
        "Heikin Ashi": 8,
        "Renko": 4,
        "Point and Figure": 6,
    }
    style = styles[
        st.radio(
            "Bar's Style",
            styles,
            index=1,
            horizontal=True,
        )
    ]  # type: ignore
    cols = st.columns(2)
    kwargs = {
        "height": 400,
        "interval": interval,
        "hide_side_toolbar": True,
        "style": style,
    }
    with cols[0]:
        trading_view(
            "SPX500USD", "OANDA", **kwargs
        )  # SP:SPX only avail from tradingview
        trading_view("GOLD", "TVC", **kwargs)
    with cols[1]:
        trading_view("DXY", "INDEX", **kwargs)
        trading_view("BLX", "BNC", **kwargs)


if __name__ == "__main__":
    set_page_config_once()
    main()

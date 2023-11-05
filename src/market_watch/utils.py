import gzip
import json
from pathlib import Path
from string import Template

import orjson
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models

from market_watch import yahoo_finance

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent.parent / "data"

TICKERS = {"BTC": "BTC-USD", "SPX": "^SPX", "GOLD": "GC=F", "DXY": "DX-Y.NYB"}


def set_page_config_once():
    try:
        st.set_page_config(
            page_title="Market Watch",
            page_icon="📈",
            layout="wide",
        )
    except Exception:
        return


def trading_view(
    name: str,
    exchange: str,
    height: int = 600,
    hide_side_toolbar: bool = False,
    interval: str = "W",
    style: int = 1,
):
    exchange_mapping = {
        "NMS": "NASDAQ",
        "NYQ": "NYSE",
        "NGM": "NASDAQ",  # Enphase
        "BTS": "AMEX",  # CBOE
    }
    exchange_long = exchange_mapping.get(exchange, exchange)
    name = name.upper().replace("-", ".")  # e.g. BRK-B
    symbol = f"{exchange_long}:{name}"
    chart = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div id="tradingview_27074"></div>
      <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/chart/?symbol=$symbol" rel="noopener nofollow" target="_blank"><span class="blue-text">$symbol</span></a></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
         new TradingView.widget(
          {
              "width": "100%",
              "height": $height,
              "symbol": "$symbol",
              "interval": "$interval",
              "timezone": "Etc/UTC",
              "theme": "dark",
              "style": "$style",
              "locale": "en",
              "enable_publishing": false,
              "withdateranges": true,
              "hide_side_toolbar": $hide_side_toolbar,
              "allow_symbol_change": true,
              "details": true,
              "container_id": "tradingview_27074"
          }
         );
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    return components.html(
        Template(chart).substitute(
            symbol=symbol,
            height=height,
            interval=interval,
            style=style,
            hide_side_toolbar=json.dumps(hide_side_toolbar),
        ),
        height=height,
    )


@st.cache_data
def get_spx_tickers_info() -> dict:
    with open(DATA_DIR / "spx_info.json.gz", "rb") as f:
        return orjson.loads(gzip.decompress(f.read()))


@st.cache_data
def get_data(symbol: str) -> dict:
    try:
        data = get_spx_tickers_info()[symbol.upper()]
    except KeyError:
        try:
            data = yahoo_finance.get_info(symbol, modules=("quoteType", "assetProfile"))
        except Exception:
            data = {}
    data = data.get("quoteType", {}) | data.get("assetProfile", {})
    return {
        col: data.get(col)
        for col in [
            "shortName",
            "longName",
            "longBusinessSummary",
            "exchange",
        ]
        if data.get(col)
    }


@st.cache_data
def get_hist(ticker: str) -> pd.DataFrame:
    try:
        df = get_spx_hists()
        data = pd.DataFrame({col: df[col][ticker] for col in ["Close", "Volume"]})
        return data
    except KeyError:
        return yf.Ticker(ticker).history(period="10y")


@st.cache_data(ttl=3600)
def get_spx_hists() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "spx_hist.parquet")


def display_tickers(names):
    df = pd.DataFrame({"LocalDate": []}).set_index("LocalDate")
    ticker_tabs = st.tabs(names)
    for i, name in enumerate(names):
        with ticker_tabs[i]:
            ticker = TICKERS.get(name, name)
            info = get_data(ticker)
            hist = get_hist(ticker)
            hist["LocalDate"] = hist.index.date
            hist = hist.set_index(["LocalDate"])
            df = df.merge(hist["Close"].rename(name), how="outer", on="LocalDate")
            st.write(
                pd.DataFrame.from_dict(
                    info,
                    orient="index",
                ).to_html(escape=False, header=False),
                unsafe_allow_html=True,
            )
            info_tabs = st.tabs(["Closing Price", "Volume", "TradingView"])
            with info_tabs[0]:
                st.plotly_chart(px.line(hist, y="Close"))
            with info_tabs[1]:
                st.plotly_chart(px.line(hist, y="Volume"))
            with info_tabs[2]:
                trading_view(name, info["exchange"])
    price_tabs = st.tabs(["closing prices", "returns data", "returns chart"])
    with price_tabs[0]:
        st.dataframe(df)
    with price_tabs[1]:
        st.dataframe((df.ffill().pct_change() + 1).cumprod())
    with price_tabs[2]:
        st.line_chart((df.ffill().pct_change() + 1).cumprod())
    mu = expected_returns.mean_historical_return(df)
    st.markdown("expected returns")
    st.dataframe(mu)
    S = risk_models.sample_cov(df)
    ef = EfficientFrontier(mu, S)
    cov = pd.DataFrame(
        ef.cov_matrix,
        index=ef.tickers,
        columns=ef.tickers,
    )
    heatmap_cov = px.imshow(cov)
    heatmap_cov.update_xaxes(side="top")
    st.text("covariance")
    st.plotly_chart(heatmap_cov)

    heatmap_cor = px.imshow(risk_models.cov_to_corr(cov))
    heatmap_cor.update_xaxes(side="top")
    st.text("correlation")
    st.plotly_chart(heatmap_cor)

    ef.max_sharpe()
    clean_weights = ef.clean_weights()
    st.plotly_chart(
        px.pie(names=list(clean_weights.keys()), values=clean_weights.values())
    )

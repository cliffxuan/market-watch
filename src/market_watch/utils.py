import json
from pathlib import Path
from string import Template

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent.parent / "data"

TICKERS = {"BTC": "BTC-USD", "SPX": "^SPX"}


def set_page_config_once():
    try:
        st.set_page_config(
            page_title="Market Watch",
            page_icon="📈",
            layout="wide",
        )
    except Exception:
        return


def trading_view(name: str, exchange: str):
    exchange_mapping = {
        "NMS": "NASDAQ",
        "NYQ": "NYSE",
        "NGM": "NASDAQ",  # Enphase
        "BTS": "AMEX",  # CBOE
    }
    exchange_long = exchange_mapping[exchange]
    name = name.upper().replace("-", ".")  # e.g. BRK-B
    symbol = f"{exchange_long}:{name}"
    chart = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div id="tradingview_27074"></div>
      <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/chart/?symbol=$symbol" rel="noopener nofollow" target="_blank"><span class="blue-text">Open in TradingView</span></a></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
         new TradingView.widget(
          {
              "width": "100%",
              "height": 932,
              "symbol": "$symbol",
              "interval": "W",
              "timezone": "Etc/UTC",
              "theme": "dark",
              "style": "1",
              "locale": "en",
              "enable_publishing": false,
              "withdateranges": true,
              "hide_side_toolbar": false,
              "allow_symbol_change": true,
              "details": true,
              "container_id": "tradingview_27074"
          }
         );
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    return components.html(Template(chart).substitute(symbol=symbol), height=1024)


@st.cache_data
def get_data(symbol: str) -> dict:
    with open(DATA_DIR / "tickers" / f"{symbol.upper()}.json") as f:
        data = json.load(f)
        return {
            col: data.get(col, "")
            for col in [
                "shortName",
                "longName",
                "longBusinessSummary",
                "exchange",
            ]
        }


@st.cache_data
def get_hist(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period="5y")


def data(names):
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
                st.line_chart(hist.Close)
            with info_tabs[1]:
                st.line_chart(hist.Volume)
            with info_tabs[2]:
                trading_view(name, info["exchange"])
    st.dataframe(df)
    mu = expected_returns.mean_historical_return(df)
    st.text("expected return")
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

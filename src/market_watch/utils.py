import json
import platform
from functools import reduce
from hashlib import sha256
from pathlib import Path
from string import Template

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns

from market_watch import ticker_data, yahoo_finance

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent.parent / "data"

TICKERS = {"BTC": "BTC-USD", "SPX": "^SPX", "GOLD": "GC=F", "DXY": "DX-Y.NYB"}


def is_local_run() -> bool:
    return bool(platform.processor().strip())


def set_page_config_once() -> None:
    try:
        st.set_page_config(
            page_title="Market Watch",
            page_icon="📈",
            layout="wide",
        )
    except Exception:
        return


@st.dialog("passkey")
def check_passkey() -> None:
    text = st.text_input("passkey", type="password")
    if st.button("Submit"):
        st.session_state.authorised = (
            sha256(text.encode("utf-8")).hexdigest() == st.secrets["passkey"]
        )
        st.rerun()


def is_authorised() -> bool:
    if is_local_run() or st.session_state.get("authorised", False):
        return True
    check_passkey()
    return False


def auth_required(f):
    def wrapped(*args, **kw):
        if is_authorised():
            return f(*args, **kw)
        else:
            st.error("refresh and enter the correct auth key")

    return wrapped


def trading_view(
    name: str,
    exchange: str,
    height: int = 600,
    hide_side_toolbar: bool = False,
    interval: str = "W",
    style: int = 1,
) -> None:
    """
    https://www.tradingview.com/widget/advanced-chart/
    """
    exchange_mapping = {
        "NMS": "NASDAQ",
        "NYQ": "NYSE",
        "NGM": "NASDAQ",  # Enphase
        "BTS": "AMEX",  # CBOE
        "NCM": "BATS",  # MARA, CLSK
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


@st.cache_data(ttl="1h")
def get_tickers_info() -> dict:
    return ticker_data.get_tickers_info()


@st.cache_data(ttl="1h")
def get_data(symbol: str) -> dict:
    try:
        data = get_tickers_info()["data"][symbol.upper()]
    except KeyError:
        try:
            data = yahoo_finance.get_info(
                symbol, modules=("quoteType", "assetProfile", "price")
            )
        except Exception:
            data = {}
    return {
        label: val
        for label, col in {
            "name": "quoteType.longName",
            "description": "assetProfile.longBusinessSummary",
            "exchange": "quoteType.exchange",
            "market cap": "price.marketCap.fmt",
        }.items()
        if (val := reduce(lambda d, k: d.get(k), col.split("."), data))
    }


@st.cache_data(ttl="1h")
def get_hist(ticker: str) -> pd.DataFrame:
    try:
        df = get_tickers_hist()
        hist = pd.DataFrame({col: df[col][ticker] for col in ["Close", "Volume"]})
    except KeyError:
        try:
            hist = yf.Ticker(ticker).history(period="10y", raise_errors=True)
        except yf.exceptions.YFInvalidPeriodError:  # type: ignore
            hist = yf.Ticker(ticker).history(period="max")
    hist["LocalDate"] = hist.index.date
    return hist.set_index(["LocalDate"])


@st.cache_data(ttl="1h")
def get_tickers_hist() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "hist.parquet")


def display_tickers(names: list[str], show_details: bool = True, optimize: bool = True):
    df = pd.DataFrame({"LocalDate": []}).set_index("LocalDate")
    for name in names:
        ticker = TICKERS.get(name, name)
        hist = get_hist(ticker)
        df = df.merge(hist["Close"].rename(name), how="outer", on="LocalDate")
    df = df.sort_index()
    if show_details:
        st.divider()
        st.markdown("## Chosen Items")
        ticker_tabs = st.tabs(names)
        for i, name in enumerate(names):
            with ticker_tabs[i]:
                ticker = TICKERS.get(name, name)
                info = get_data(ticker)
                st.write(
                    pd.DataFrame.from_dict(
                        info,
                        orient="index",
                    ).to_html(escape=False, header=False),
                    unsafe_allow_html=True,
                )
                info_tabs = st.tabs(["TradingView", "Closing Price", "Volume"])
                with info_tabs[0]:
                    trading_view(name, info["exchange"])
                with info_tabs[1]:
                    st.plotly_chart(
                        px.line(get_hist(ticker), y="Volume"), use_container_width=True
                    )
                with info_tabs[2]:
                    st.plotly_chart(
                        px.line(get_hist(ticker), y="Close"), use_container_width=True
                    )
    st.divider()
    st.markdown("## Collective")
    price_tabs = st.tabs(
        [
            "cumulative return chart",
            "daily close",
            "cumulative return data",
        ]
    )
    with price_tabs[0]:
        st.plotly_chart(
            px.line((df.ffill().pct_change() + 1).cumprod(), log_y=True),
            use_container_width=True,
        )
    with price_tabs[1]:
        st.dataframe(df.sort_index(ascending=False))
    with price_tabs[2]:
        st.dataframe(
            (df.ffill().pct_change() + 1).cumprod().sort_index(ascending=False)
        )

    if len(names) < 2:
        return
    cov = df.pct_change(fill_method=None).dropna(how="all").cov() * 252
    corr_heatmap = px.imshow(df.corr(), text_auto=True)
    corr_heatmap.update_xaxes(side="top")
    cov_heatmap = px.imshow(cov, text_auto=True)
    cov_heatmap.update_xaxes(side="top")
    st.divider()
    cov_cor_cols = st.columns(2)
    with cov_cor_cols[0]:
        st.markdown("### Correlation")
        corr_tabs = st.tabs(["heatmap", "data"])
        with corr_tabs[0]:
            st.plotly_chart(corr_heatmap, theme=None)
        with corr_tabs[1]:
            st.dataframe(df.corr())

    with cov_cor_cols[1]:
        st.markdown("### Covariance")
        cov_tabs = st.tabs(["heatmap", "data"])
        with cov_tabs[0]:
            st.plotly_chart(cov_heatmap, theme=None)
        with cov_tabs[1]:
            st.dataframe(cov)

    if optimize:
        st.markdown("### Portfolio Optimisation")
        mu: pd.Series = expected_returns.mean_historical_return(df)
        returns_df = pd.DataFrame(
            {"mean historical return": mu, "expected return": round(mu, 4)},
        )
        returns_df.index.name = "ticker"
        st.divider()
        st.markdown("#### Expected Returns")
        returns_df = st.data_editor(
            returns_df, disabled=["mean historical return", "ticker"]
        )
        returns = returns_df["expected return"]
        ef = EfficientFrontier(returns, cov)

        st.divider()
        st.markdown("#### Optimized Portfolios")
        st.markdown("##### Maximize Sharpe Ratio")
        risk_free_rate = st.slider(
            "risk free rate",
            min_value=0.005,
            max_value=0.1,
            step=0.005,
            value=0.02,
            format="%.3f",
        )
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        clean_weights = ef.clean_weights()
        max_sharpe_cols = st.columns([2, 1])
        with max_sharpe_cols[0]:
            st.plotly_chart(
                px.pie(names=list(clean_weights.keys()), values=clean_weights.values())
            )
        ws = np.array(list(clean_weights.values()))
        with max_sharpe_cols[1]:
            st.write(pd.Series(ws, cov.index, name="weight"))
            st.metric("risk", f"{np.sqrt(ws @ cov @ ws):.3f}")
            st.metric("return", f"{ws @ returns:.3f}")
        st.divider()
        st.markdown("##### Minimize Risk")
        w = cp.Variable(len(returns))
        min_var_prob = cp.Problem(
            objective=cp.Minimize(cp.QuadForm(w, cov)),
            constraints=[
                cp.sum(w) == 1,
                w >= 0,
            ],
        )
        min_var_prob.solve()
        min_risk_cols = st.columns([2, 1])
        with min_risk_cols[0]:
            st.plotly_chart(px.pie(names=cov.index, values=w.value))
        with min_risk_cols[1]:
            st.write(pd.Series(w.value, cov.index, name="weight"))
            st.metric("risk", f"{np.sqrt(w.value @ cov @ w.value):.3f}")
            st.metric("return", f"{w.value @ returns:.3f}")
        st.divider()
        st.markdown("##### Efficient Frontier")
        solutions = []
        for ret in np.linspace(returns.min(), returns.max(), 20):
            prob = cp.Problem(
                objective=cp.Minimize(cp.QuadForm(w, cov)),
                constraints=[
                    cp.sum(w) == 1,
                    w >= 0,
                    w @ returns == ret,
                ],
            )
            prob.solve()
            solutions.append(
                {
                    "return": ret,
                    "risk": np.sqrt(w.value @ cov @ w.value),
                }
            )
        ef_cols = st.columns([2, 1])
        with ef_cols[0]:
            st.plotly_chart(px.scatter(pd.DataFrame(solutions), x="risk", y="return"))
        with ef_cols[1]:
            st.dataframe(pd.DataFrame(solutions))

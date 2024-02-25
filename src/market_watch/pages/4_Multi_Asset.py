import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from market_watch.utils import DATA_DIR, set_page_config_once

if __name__ == "__main__":
    set_page_config_once()
    st.markdown("## Long-Term Capital Market Assumptions")
    st.divider()
    st.markdown("### Risk & Return")
    risk_return = pd.read_csv(
        DATA_DIR / "multi-asset-risk-return.csv", index_col="Asset Class"
    )
    st.dataframe(
        risk_return,
        height=960,
        column_order=[
            "Asset Class",
            "Asset Class Type",
            "Volatility",
            "Compund Return 2024",
            "Arithmetic Return 2024",
            "Compund Return 2023",
        ],
    )
    fig = px.scatter(
        risk_return,
        x="Volatility",
        y="Compund Return 2024",
        color=risk_return.index,
        symbol="Asset Class Type",
    )
    fig.update_traces(marker={"size": 8})
    st.plotly_chart(
        fig,
        use_container_width=True,
    )
    st.divider()
    st.markdown("### Correlation Matrices")
    corr = pd.read_csv(
        DATA_DIR / "multi-asset-correlation.csv", index_col="Asset Class"
    )
    st.dataframe(corr, height=960)
    components.html(
        '<a href="https://am.jpmorgan.com/us/en/asset-management/institutional/insights/portfolio-insights/ltcma/interactive-assumptions-matrices/" target="_">data source</a>'
    )
    corr_heatmap = px.imshow(corr, text_auto=True)
    st.plotly_chart(corr_heatmap, theme=None, use_container_width=True)

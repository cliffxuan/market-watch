import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from market_watch.utils import DATA_DIR, set_page_config_once

if __name__ == "__main__":
    set_page_config_once()
    st.markdown("## Long-Term Capital Market Assumptions Correlation Matrices")
    corr = pd.read_csv(
        DATA_DIR / "multi-asset-correlation.csv", index_col="Asset Class"
    )
    st.dataframe(corr, height=960)
    components.html(
        '<a href="https://am.jpmorgan.com/us/en/asset-management/institutional/insights/portfolio-insights/ltcma/interactive-assumptions-matrices/" target="_">data source</a>'
    )
    corr_heatmap = px.imshow(corr, text_auto=True)
    st.plotly_chart(corr_heatmap, theme=None, use_container_width=True)

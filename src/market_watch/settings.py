import streamlit as st

GPT_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = st.secrets["openai_api_key"]
# https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas?project=cliffxuan
YOUTUBE_API_KEY = st.secrets["youtube_api_key"]
YOUTUBE_TRANSCRIPT_API_PROXY = st.secrets.get("youtube_transcript_api_proxy")

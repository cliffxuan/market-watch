from __future__ import annotations

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build

from market_watch.index_table import search
from market_watch.settings import YOUTUBE_API_KEY
from market_watch.utils import auth_required, set_page_config_once
from market_watch.youtube import CHANNELS, Video


@st.cache_data(ttl="2h")
def get_latest_videos(channel_id: str, limit: int = 5) -> list[dict]:
    """
    https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas
    """
    return (
        build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        .search()
        .list(
            channelId=channel_id,
            part="id,snippet",
            order="date",
            type="video",
            maxResults=limit,
        )
        .execute()
    ).get("items", [])


@auth_required
def main() -> None:
    st.markdown("# Influencers")
    # Process each channel
    all_data = []
    all_videos = {}
    for channel, channel_id in CHANNELS.items():
        with st.spinner(f"\nAnalyzing channel: {channel}"):
            data = get_latest_videos(channel_id, limit=5)
            all_data = [*all_data, *data]
            for video in Video.process_videos(data):
                all_videos[video.id] = video

    all_videos_df = pd.DataFrame(
        [video.model_dump() for video in all_videos.values()]
    ).sort_values(by="publish_time", ascending=False)
    all_videos_df["select"] = st.session_state.setdefault(
        f"{__name__}.selected.{len(all_videos)}", [False] * len(all_videos)
    )
    all_videos_df["publish_time"] = pd.to_datetime(
        all_videos_df["publish_time"]
    ).dt.strftime("%Y-%m-%d %H:%M")
    cols = [
        "thumbnail_url",
        "channel_url",
        "publish_time",
        "title",
        "summary",
    ]
    query = st.text_input(
        "search",
        label_visibility="collapsed",
        placeholder="search",
    )
    all_videos_df = search(all_videos_df, query)
    tabs = st.tabs(["Table", "Data"])
    with tabs[0]:
        selected = st.data_editor(
            all_videos_df,
            column_order=["select", *cols],
            column_config={
                "select": st.column_config.CheckboxColumn(
                    "",
                    help="Select to see more info and portfolio optimization",
                    default=False,
                ),
                "channel_url": st.column_config.LinkColumn(
                    "channel",
                    validate=r"^https://youtube\.com/@(.+)$",
                    max_chars=100,
                    display_text=r"^https://youtube\.com/@(.+)$",
                ),
                "thumbnail_url": st.column_config.ImageColumn("thumbnail"),
            },
            disabled=cols,
            hide_index=True,
            height=(len(all_videos_df) + 1) * 35 + 3,
        )
        ids = list(selected[selected["select"]]["id"])
        for video_id in ids:
            video = all_videos[video_id]
            video.render()
            st.divider()

    with tabs[1]:
        with st.expander("raw"):
            st.json(all_data)
        with st.expander("videos"):
            st.json([video.model_dump() for video in all_videos.values()])


if __name__ == "__main__":
    set_page_config_once()
    main()

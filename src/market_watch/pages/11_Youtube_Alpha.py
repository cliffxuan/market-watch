from __future__ import annotations

import streamlit as st

from market_watch.settings import YOUTUBE_API_KEY
from market_watch.utils import auth_required, set_page_config_once
from market_watch.youtube import Video, get_video_id, get_video_metadata


@auth_required
def main() -> None:
    st.markdown("# Youtube Alpha")
    video_url = st.text_input("video url")
    if not video_url:
        return
    video_id = get_video_id(video_url)
    if not video_id:
        st.error("invalid url")
        return
    with st.spinner("Processing..."):
        metadata = get_video_metadata(video_id, YOUTUBE_API_KEY)
    with st.spinner("Sumarizing..."):
        video = Video.process_video(metadata)
    video.render()


if __name__ == "__main__":
    set_page_config_once()
    main()

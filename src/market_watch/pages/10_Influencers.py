from __future__ import annotations

import datetime as dt
import html
import json
import logging
import textwrap
from functools import cached_property
from hashlib import md5
from pathlib import Path
from typing import Generator

import openai
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from pydantic import BaseModel, computed_field
from youtube_transcript_api import YouTubeTranscriptApi

from market_watch.utils import auth_required, set_page_config_once

# https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas?project=cliffxuan
YOUTUBE_API_KEY = st.secrets["youtube_api_key"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
YOUTUBE_TRANSCRIPT_API_PROXY = st.secrets.get("youtube_transcript_api_proxy")

CHANNELS = {
    "@CoinBureau": "UCqK_GSMbpiV8spgD3ZGloSw",
    "@intothecryptoverse": "UCRvqjQPSeaWn-uEx-w0XOIg",
    "@AlexBeckersChannel": "UCKQvGU-qtjEthINeViNbn6A",
    "@AltcoinDaily": "UCbLhGKVY-bJPcawebgtNfbw",
    "@VirtualBacon": "UCcrEA_xd9Ldf1C8DIJYdyyA",
    "@Jungernaut": "UCQglaVhGOBI0BR5S6IJnQPg",
    "@TheMoon": "UCc4Rz_T9Sb1w5rqqo9pL1Og",
    "@TokenMetrics": "UCH9MOLQ_KUpZ_cw8uLGUisA",
    "@CryptoBanterGroup": "UCN9Nj4tjXbVTLYWN0EKly_Q",
    "@Coinsider": "UCi7egjf0JDHuhznWugXq4hA",
    "@DataDash": "UCCatR7nWbYrkVXdxXb4cGXw",
}
CHANNELS_REVERSE = {v: k for k, v in CHANNELS.items()}


PWD = Path(__file__).absolute().parent
STORE = PWD / ".store"
STORE.mkdir(exist_ok=True)


def md_escape(text: str, multi_line: bool = False) -> str:
    text = text.replace("$", "\\$")
    if multi_line:
        text = text.replace(". ", ".\n\n")
    return text


class Video(BaseModel):
    id: str
    title: str
    description: str
    channel: str
    publish_time: dt.datetime
    thumbnail_url: str

    @computed_field
    @cached_property
    def captions(self) -> str | None:
        return get_video_captions(self.id)

    @computed_field
    @cached_property
    def summary(self) -> str | None:
        if self.captions is not None:
            return summarize_text(
                "\n".join([self.title, self.description, self.captions])
            )

    @computed_field
    def url(self) -> str:
        return f"https://youtube.com/watch?v={self.id}"

    @computed_field
    def channel_url(self) -> str:
        return f"https://youtube.com/{self.channel}"

    @classmethod
    def process_videos(cls, data: list[dict]) -> Generator[Video, None, None]:
        for item in data:
            video = cls.model_validate(
                {
                    "id": item["id"]["videoId"],
                    "title": html.unescape(item["snippet"]["title"]),
                    "description": item["snippet"]["description"],
                    "thumbnail_url": item["snippet"]["thumbnails"]["default"]["url"],
                    "channel": CHANNELS_REVERSE[item["snippet"]["channelId"]],
                    "publish_time": item["snippet"]["publishTime"],
                }
            )
            if video.summary is not None:
                logging.info("succeed getting summary for vidoe")
            yield video


def get_video_captions(video_id: str) -> str | None:
    """
    Retrieve captions for a given video

    :param video_id: YouTube video ID
    :return: Full caption text or None
    """
    file_path = STORE / "captions.json"
    try:
        with file_path.open("r") as f:
            all_captions = json.loads(f.read())
    except FileNotFoundError:
        all_captions = {}
    if video_id in all_captions:
        return all_captions[video_id]
    try:
        if YOUTUBE_TRANSCRIPT_API_PROXY is None:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, proxies={"https": YOUTUBE_TRANSCRIPT_API_PROXY}
            )
        captions = " ".join([entry["text"] for entry in transcript])
        all_captions[video_id] = captions
        with file_path.open("w") as f:
            f.write(json.dumps(all_captions))
        return captions
    except Exception as e:
        print(f"Could not retrieve captions for {video_id}: {e}")
        return None


@st.cache_data(ttl="2h")
def get_latest_videos(channel_id: str, limit: int = 5) -> list[dict]:
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


def summarize_text(text: str, max_length: int = 1_000) -> str:
    """
    Summarize text using OpenAI API

    :param text: Text to summarize
    :param max_length: Maximum summary length
    :return: Summary of the text
    """
    hash_key = md5(text.encode("utf-8")).hexdigest()
    file_path = STORE / "summary.json"
    try:
        with file_path.open("r") as f:
            all_summaries = json.loads(f.read())
    except FileNotFoundError:
        all_summaries = {}
    if hash_key in all_summaries:
        return all_summaries[hash_key]
    try:
        response = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a helpful assistant that summarizes content concisely.
                    """,
                },
                {
                    "role": "user",
                    "content": textwrap.dedent(
                        f"""
                    Provide a concise summary of the following text
                    (aim for {max_length} characters): {text}
                    """.strip()
                    ),
                },
            ],
            max_tokens=max_tokens_for_summary(text),
        )
        summary = response.choices[0].message.content.strip()  # type: ignore
        all_summaries[hash_key] = summary
        with file_path.open("w") as f:
            f.write(json.dumps(all_summaries))
        return summary
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Could not generate summary."


def max_tokens_for_summary(text: str) -> int:
    """
    Calculate appropriate max tokens for summarization based on input text length
    """
    return min(max(100, len(text) // 4), 1000)


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
    cols = [
        "thumbnail_url",
        "title",
        "channel_url",
        "publish_time",
        "summary",
    ]
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
            height=(len(all_videos) + 1) * 35 + 3,
        )
        ids = list(selected[selected["select"]]["id"])
        for video_id in ids:
            video = all_videos[video_id]
            st.markdown(f"- {video.title}")
            st.markdown(f"[![{video.url}]({video.thumbnail_url})]({video.url})")
            st.markdown(f"Channel: [{video.channel}]({video.channel_url})")
            st.markdown(f"Description: {md_escape(video.description)}")
            st.markdown(f"Publish Time: {video.publish_time}")
            if video.captions is not None:
                st.markdown(f"Summary: {md_escape(video.summary, multi_line=True)}")
            else:
                st.error("No captions available")
            with st.expander("data"):
                st.json(all_videos[video_id].model_dump())
            st.divider()

    with tabs[1]:
        with st.expander("raw"):
            st.json(all_data)
        with st.expander("videos"):
            st.json([video.model_dump() for video in all_videos.values()])


if __name__ == "__main__":
    set_page_config_once()
    main()

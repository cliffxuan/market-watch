from __future__ import annotations

import datetime as dt
import html
import json
import logging
import re
import textwrap
from functools import cached_property
from hashlib import md5
from pathlib import Path
from typing import Generator

import openai
import streamlit as st
from googleapiclient.discovery import build
from pydantic import BaseModel, computed_field
from youtube_transcript_api import YouTubeTranscriptApi

from market_watch.open_ai import summarize
from market_watch.settings import (
    GPT_MODEL,
    OPENAI_API_KEY,
    YOUTUBE_API_KEY,
    YOUTUBE_TRANSCRIPT_API_PROXY,
)

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
    "@Bankless": "UCAl9Ld79qaZxp9JzEOwd3aA",
}
CHANNELS_REVERSE = {v: k for k, v in CHANNELS.items()}

PWD = Path(__file__).absolute().parent
STORE = PWD.parent.parent / ".store"
STORE.mkdir(exist_ok=True)


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
        return get_video_captions(self.id, YOUTUBE_TRANSCRIPT_API_PROXY)

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
    def process_video(cls, item: dict) -> Video:
        video = cls.model_validate(
            {
                "id": (
                    item["id"]["videoId"]
                    if isinstance(item["id"], dict)
                    else item["id"]
                ),
                "title": html.unescape(item["snippet"]["title"]),
                "description": item["snippet"]["description"],
                "thumbnail_url": item["snippet"]["thumbnails"]["default"]["url"],
                "channel": get_channel(item["snippet"]["channelId"], YOUTUBE_API_KEY),
                "publish_time": item["snippet"]["publishedAt"],
            }
        )
        if video.summary is not None:
            logging.info("succeed getting summary for vidoe")
        return video

    @classmethod
    def process_videos(cls, data: list[dict]) -> Generator[Video, None, None]:
        for item in data:
            yield cls.process_video(item)

    def render(self) -> None:
        st.markdown(f"- {self.title}")
        st.markdown(f"[![{self.url}]({self.thumbnail_url})]({self.url})")
        st.markdown(f"Channel: [{self.channel}]({self.channel_url})")
        st.markdown(f"Description: {escape_markdown(self.description)}")
        st.markdown(f"Publish Time: {self.publish_time}")
        if self.summary is not None:
            st.markdown(self.summary)
        else:
            st.error("No captions available")
        if prompt := st.chat_input("ask some question", key=f"chat_input.{self.id}"):
            messages = st.container(height=300)
            messages.chat_message("user").write(prompt)
            response = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """
            You are a helpful assistant that answers questions precisely.
            """,
                    },
                    {
                        "role": "user",
                        "content": textwrap.dedent(
                            f"""
            answer question "{prompt}"
            context: {self.model_dump_json()}
            """.strip()
                        ),
                    },
                ],
            )
            messages.chat_message("assistant").write(
                response.choices[0].message.content.strip()
            )

        with st.expander("data"):
            st.json(self.model_dump())


def get_channel(channel_id: str, developer_key: str) -> str:
    try:
        return CHANNELS_REVERSE[channel_id]
    except KeyError:
        pass
    youtube = build("youtube", "v3", developerKey=developer_key)
    channel_response = youtube.channels().list(part="snippet", id=channel_id).execute()
    channel = channel_response["items"][0]
    custom_url = channel["snippet"].get("customUrl", "")
    return custom_url if custom_url.startswith("@") else f"@{custom_url}"


def max_tokens_for_summary(text: str) -> int:
    """
    Calculate appropriate max tokens for summarization based on input text length
    """
    return min(max(100, len(text) // 4), 1000)


def get_video_captions(video_id: str, proxy: str | None = None) -> str | None:
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
        if proxy is None:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, proxies={"https": proxy}
            )
        captions = " ".join([entry["text"] for entry in transcript])
        all_captions[video_id] = captions
        with file_path.open("w") as f:
            f.write(json.dumps(all_captions))
        return captions
    except Exception as e:
        print(f"Could not retrieve captions for {video_id}: {e}")
        return None


def summarize_text(text: str) -> str:
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
        summary = summarize(text)
        all_summaries[hash_key] = summary
        with file_path.open("w") as f:
            f.write(json.dumps(all_summaries))
        return summary
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Could not generate summary."


def get_transcript(
    video_id: str, youtube_transcript_api_proxy: str | None = None
) -> list[dict]:
    try:
        if youtube_transcript_api_proxy is None:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, proxies={"https": youtube_transcript_api_proxy}
            )
        return transcript
    except Exception as e:
        print(f"Could not retrieve captions for {video_id}: {e}")
        return []


def get_video_id(url: str) -> str | None:
    # Handle multiple URL formats
    patterns = [
        r"(?:v=|/)([\w-]{11})(?:\S+)?$",  # Standard and embed URLs
        r"(?:youtu\.be/)([\w-]{11})",  # Short URLs
        r"(?:shorts/)([\w-]{11})",  # Shorts URLs
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_video_metadata(video_id: str, developer_key: str) -> dict:
    youtube = build("youtube", "v3", developerKey=developer_key)
    return (
        youtube.videos()
        .list(
            part="id,snippet,statistics",
            id=video_id,
        )
        .execute()
    ).get("items", [])[0]


def escape_markdown(text: str) -> str:
    """Escape special characters for Markdown V2."""
    special_chars = [
        "_",
        "*",
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text

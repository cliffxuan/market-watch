from __future__ import annotations

import datetime as dt
import html
import re
from pathlib import Path
from typing import ClassVar, Generator, Optional

import streamlit as st
from googleapiclient.discovery import build
from pydantic import computed_field
from sqlmodel import Field, Session, SQLModel, create_engine
from youtube_transcript_api import YouTubeTranscriptApi

from market_watch.open_ai import answer_transcript_question, summarize
from market_watch.settings import (
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
DATA_DIR = PWD.parent.parent / "data"


class Video(SQLModel, table=True):
    """YouTube video model with SQLite storage."""

    # Allow table redefinition
    __table_args__: ClassVar = {"extend_existing": True}

    # Required fields
    id: str = Field(primary_key=True)
    title: str
    description: str
    channel: str
    publish_time: dt.datetime
    thumbnail_url: str

    # Optional cached fields
    captions: Optional[str] = Field(default=None)
    summary: Optional[str] = Field(default=None)

    # Computed fields (not stored in DB)
    @computed_field
    def url(self) -> str:
        return f"https://youtube.com/watch?v={self.id}"

    @computed_field
    def channel_url(self) -> str:
        return f"https://youtube.com/{self.channel}"

    @classmethod
    def process_video(cls, item: dict) -> "Video":
        """Create or update a Video instance from YouTube API response."""
        video_id = item["id"]["videoId"] if isinstance(item["id"], dict) else item["id"]

        with Session(engine) as session:
            # Try to get existing video
            video = session.get(Video, video_id)

            if video is None:
                # Create new video
                # Convert publish_time string to datetime object
                publish_time = dt.datetime.fromisoformat(
                    item["snippet"]["publishedAt"].replace("Z", "+00:00")
                )

                video = cls(
                    id=video_id,
                    title=html.unescape(item["snippet"]["title"]),
                    description=item["snippet"]["description"],
                    thumbnail_url=item["snippet"]["thumbnails"]["default"]["url"],
                    channel=get_channel(item["snippet"]["channelId"], YOUTUBE_API_KEY),
                    publish_time=publish_time,  # Now using datetime object
                )
                session.add(video)

            # Update captions if not already cached
            if video.captions is None:
                video.captions = get_video_captions(
                    video_id, YOUTUBE_TRANSCRIPT_API_PROXY
                )

            # Update summary if we have captions but no summary
            if video.captions and video.summary is None:
                video.summary = summarize(
                    "\n".join([video.title, video.description, video.captions])
                )

            session.commit()
            session.refresh(video)

            return video

    @classmethod
    def process_videos(cls, data: list[dict]) -> Generator[Video, None, None]:
        for item in data:
            yield cls.process_video(item)

    def render(self) -> None:
        st.text(f"- {self.title}")
        st.markdown(f"[![{self.url}]({self.thumbnail_url})]({self.url})")
        st.markdown(f"Channel: [{self.channel}]({self.channel_url})")
        st.text(f"Description: {self.description}")
        st.text(f"Publish Time: {self.publish_time}")

        if self.summary:
            st.text(self.summary)
        else:
            st.error("No captions available")

        # Chat interface
        if self.captions:  # Only show chat if we have captions
            st.markdown("---")
            st.markdown("💬 **Ask questions about the video**")

            if prompt := st.chat_input(
                "Ask a question about the video content...",
                key=f"chat_input.{self.id}",
            ):
                messages = st.container(height=300)
                messages.chat_message("user").write(prompt)

                # Use the dedicated question answering function
                answer = answer_transcript_question(
                    "\n".join(
                        [
                            f"Title: {self.title}",
                            f"Description: {self.description}",
                            f"Transcript: {self.captions}",
                        ]
                    ),
                    prompt,
                )

                messages.chat_message("assistant").write(answer)
        else:
            st.info("Chat is unavailable without video captions")

        # Debug data
        with st.expander("Debug Data"):
            st.json(
                self.model_dump(
                    include={"id", "title", "channel", "publish_time", "url"}
                )
            )


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
    """Retrieve captions for a given video."""
    try:
        if proxy is None:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, proxies={"https": proxy}
            )
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        print(f"Could not retrieve captions for {video_id}: {e}")
        return None


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
    """
    https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas
    """
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


# Setup database
engine = create_engine(f"sqlite:///{DATA_DIR}/db.sqlite3", echo=True)


def init_db() -> None:
    """Initialize database and create tables."""
    # Clear any existing table definitions
    # SQLModel.metadata.clear()
    # Create all tables if they don't exist
    SQLModel.metadata.create_all(engine)


# Initialize database on module load
init_db()

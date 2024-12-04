import json
import textwrap
from hashlib import md5
from pathlib import Path

import openai
import streamlit as st
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

from market_watch.utils import is_authorised, set_page_config_once

# https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas?project=cliffxuan
YOUTUBE_API_KEY = st.secrets["youtube_api_key"]
OPENAI_API_KEY = st.secrets["openai_api_key"]

CHANNELS = {
    "@CoinBureau": "UCqK_GSMbpiV8spgD3ZGloSw",
    "@intothecryptoverse": "UCRvqjQPSeaWn-uEx-w0XOIg",
    "@AlexBeckersChannel": "UCKQvGU-qtjEthINeViNbn6A",
    "@AltcoinDaily": "UCbLhGKVY-bJPcawebgtNfbw",
    "@VirtualBacon": "UCcrEA_xd9Ldf1C8DIJYdyyA",
    "@Jungernaut": "UCQglaVhGOBI0BR5S6IJnQPg",
}


PWD = Path(__file__).absolute().parent
STORE = PWD / ".store"
STORE.mkdir(exist_ok=True)


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
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        captions = " ".join([entry["text"] for entry in transcript])
        all_captions[video_id] = captions
        with file_path.open("w") as f:
            f.write(json.dumps(all_captions))
        return captions
    except Exception as e:
        print(f"Could not retrieve captions for {video_id}: {e}")
        return None


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


def main() -> None:
    st.markdown("# Influencers")
    # Process each channel
    for channel, channel_id in CHANNELS.items():
        st.markdown(f"## {channel}")
        with st.spinner(f"\nAnalyzing channel: {channel}"):
            # Get latest videos
            videos = get_latest_videos(channel_id, limit=2)

            st.json(videos, expanded=False)
            for video in videos:
                video_id = video["id"]["videoId"]
                title = video["snippet"]["title"]
                description = video["snippet"]["description"]
                st.image(video["snippet"]["thumbnails"]["default"]["url"])
                st.markdown(f"Title: {title}")
                st.markdown(f"Description: {description}")
                st.markdown(f"Publish Time: {video['snippet']['publishTime']}")
                # Get captions
                captions = get_video_captions(video_id)
                if captions:
                    # Summarize captions
                    summary = summarize_text(
                        "\n".join([title, description, captions])
                    ).replace("$", "\\$")
                    st.markdown(f"Summary: {summary}")
                else:
                    st.error("No captions available")
                st.divider()


if __name__ == "__main__":
    set_page_config_once()
    if is_authorised():
        main()
    else:
        st.error("refresh and enter the correct auth key")

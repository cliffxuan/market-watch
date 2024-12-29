from __future__ import annotations

import json

import openai

from market_watch.settings import (
    GPT_MODEL,
    OPENAI_API_KEY,
)


def punctuate(transcript: list[dict]) -> str:
    response = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful assistant.
                try to put puncuation into the transcript and return text.
                """,
            },
            {
                "role": "user",
                "content": json.dumps(transcript),
            },
        ],
        max_tokens=1000 * 2,
    )
    summary = response.choices[0].message.content.strip()  # type: ignore
    return summary


def summarize(text: str) -> str:
    print("openai:", openai)
    response = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful assistant that summarizes content concisely.
                If this talks about what coins to invest, list the coins.
                Return text in markdown format.
                """,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        max_tokens=max_tokens_for_summary(text),
    )
    summary = response.choices[0].message.content.strip()  # type: ignore
    return summary


def max_tokens_for_summary(text: str) -> int:
    """
    Calculate appropriate max tokens for summarization based on input text length
    """
    return min(max(100, len(text) // 4), 1000)

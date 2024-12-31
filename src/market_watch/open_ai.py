from __future__ import annotations

import json

import openai

from market_watch.settings import (
    GPT_MODEL,
    OPENAI_API_KEY,
)

DEFAULT_SUMMARIZE_PROMPT = """
You are a helpful assistant that provides clear, structured summaries.
Format the response in three sections:
1. Start each section with "📌 Main Points:", "🔑 Key Topics:", and "📋 Summary:" respectively
2. Use simple bullet points with '•' (no nested bullets)
3. Don't use any markdown symbols like **, __, #, or other special characters
4. Keep formatting minimal and clean"""


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


def summarize(text: str, prompt: str = DEFAULT_SUMMARIZE_PROMPT) -> str:
    response = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt,
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

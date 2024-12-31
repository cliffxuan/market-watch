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
4. Keep formatting minimal and clean
"""  # noqa: E501

CRYPTO_ANALYSIS_PROMPT = """
You are a cryptocurrency analysis assistant. Review the text and extract cryptocurrency-related information.
If no cryptocurrencies are mentioned, respond with an empty string.
Otherwise, format your response with "🪙 Coins:" followed by bullet points for each cryptocurrency mentioned:
• {Coin Name}:
  - Key predictions/points
  - Price targets and timeframes
  - Context and reasoning
Keep formatting minimal and clean, using only bullet points (•) and no special characters.
"""  # noqa: E501


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
    summary = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
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
    summary_text = summary.choices[0].message.content.strip()

    # Add crypto analysis if relevant
    crypto_analysis = analyze_crypto_content(text)
    if crypto_analysis:
        summary_text = f"{summary_text}\n\n{crypto_analysis}"

    return summary_text


def max_tokens_for_summary(text: str) -> int:
    """
    Calculate appropriate max tokens for summarization based on input text length
    """
    return min(max(100, len(text) // 4), 1000)


def analyze_crypto_content(text: str) -> str:
    """Analyze text for cryptocurrency-related content."""
    response = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": CRYPTO_ANALYSIS_PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        max_tokens=500,  # Smaller token limit for focused analysis
    )
    return response.choices[0].message.content.strip()

from os import getenv
from urllib.parse import parse_qs, urlparse

from openai import OpenAI
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from youtube_transcript_api import YouTubeTranscriptApi

from market_watch.settings import (
    OPENAI_API_KEY,
    TELEGRAM_BOT_TOKEN,
    YOUTUBE_TRANSCRIPT_API_PROXY,
)

# Replace with your tokens

# Conversation states
WAITING_FOR_QUESTION = 1

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = (
        "*Welcome to YouTube Transcript Analyzer\!*\n\n"
        "Send me a YouTube URL and I'll provide:\n"
        "• A concise summary of the content\n"
        "• Ability to ask questions about the video\n\n"
        "_Just paste a YouTube URL to begin\!_"
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN_V2)


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    parsed_url = urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]
    return None


def get_transcript(video_id: str) -> str:
    """Get transcript from YouTube video."""
    try:
        https = getenv("YOUTUBE_TRANSCRIPT_API_PROXY") or YOUTUBE_TRANSCRIPT_API_PROXY
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, proxies={"https": https}
        )
        transcript_text = ""
        for segment in transcript_list:
            timestamp = int(segment["start"])
            minutes = timestamp // 60
            seconds = timestamp % 60
            transcript_text += f"[{minutes:02d}:{seconds:02d}] {segment['text']}\n"
        return transcript_text
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"


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


def get_summary(transcript: str) -> str:
    """Get summary of the transcript using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that provides clear, structured summaries. 
                Format the response in three sections:
                1. Start each section with "📌 Main Points:", "🔑 Key Topics:", and "📋 Summary:" respectively
                2. Use simple bullet points with '•' (no nested bullets)
                3. Don't use any markdown symbols like **, __, #, or other special characters
                4. Keep formatting minimal and clean""",
                },
                {
                    "role": "user",
                    "content": f"Please provide a structured summary of this transcript:\n{transcript}",
                },
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"


async def handle_youtube_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle YouTube URLs sent to the bot."""
    url = update.message.text
    video_id = extract_video_id(url)

    if not video_id:
        await update.message.reply_text("⚠️ Please send a valid YouTube URL.")
        return ConversationHandler.END

    await update.message.reply_text(
        "🔄 Processing video content\nFetching transcript and generating summary..."
    )

    transcript = get_transcript(video_id)
    context.user_data["transcript"] = transcript

    # Generate and send summary
    summary = get_summary(transcript)
    # Send without markdown parsing to preserve simple formatting
    await update.message.reply_text(summary)

    await update.message.reply_text(
        "💡 Ask Questions\n"
        "You can now ask specific questions about the content.\n"
        "Send /done when you're finished."
    )
    return WAITING_FOR_QUESTION


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user questions about the transcript."""
    question = update.message.text
    transcript = context.user_data.get("transcript", "")

    if not transcript:
        await update.message.reply_text(
            "⚠️ Error: No transcript loaded. Please send a YouTube URL first."
        )
        return ConversationHandler.END

    await update.message.reply_text("🤔 Analyzing your question...")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant answering questions about a video transcript. Base your answers strictly on the transcript content. Keep formatting simple and avoid using special characters.",
                },
                {
                    "role": "user",
                    "content": f"Here's a transcript:\n{transcript}\n\nQuestion: {question}",
                },
            ],
        )

        answer = response.choices[0].message.content
        await update.message.reply_text(f"💬 {answer}")
        return WAITING_FOR_QUESTION

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")
        return WAITING_FOR_QUESTION


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """End the conversation."""
    await update.message.reply_text(
        "✅ *Session ended*\n"
        "_Send another YouTube URL whenever you want to analyze a new video\\!_",
        parse_mode=ParseMode.MARKDOWN_V2,
    )
    context.user_data.clear()
    return ConversationHandler.END


def main() -> None:
    """Start the bot."""
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_youtube_url)
        ],
        states={
            WAITING_FOR_QUESTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question),
                CommandHandler("done", done),
            ],
        },
        fallbacks=[CommandHandler("done", done)],
    )

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(conv_handler)

    # Start the bot
    application.run_polling()


if __name__ == "__main__":
    main()

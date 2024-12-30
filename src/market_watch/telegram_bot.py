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

from market_watch.settings import OPENAI_API_KEY, TELEGRAM_BOT_TOKEN

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
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
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
                    "content": "You are a helpful assistant that provides clear, structured summaries. Format the response in sections using markdown with headings for 'Main Points', 'Key Topics', and 'Summary'.",
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
        await update.message.reply_text(
            "⚠️ Please send a valid YouTube URL\\.", parse_mode=ParseMode.MARKDOWN_V2
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "🔄 *Processing video content*\n_Fetching transcript and generating summary\\.\\.\\._",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    transcript = get_transcript(video_id)
    context.user_data["transcript"] = transcript

    # Generate and send summary
    summary = get_summary(transcript)
    summary_message = f"📋 *Video Analysis*\n\n{escape_markdown(summary)}"
    await update.message.reply_text(summary_message, parse_mode=ParseMode.MARKDOWN_V2)

    await update.message.reply_text(
        "💡 *Ask Questions*\n"
        "You can now ask specific questions about the content\\.\n"
        "_Send_ /done _when you're finished\\._",
        parse_mode=ParseMode.MARKDOWN_V2,
    )
    return WAITING_FOR_QUESTION


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user questions about the transcript."""
    question = update.message.text
    transcript = context.user_data.get("transcript", "")

    if not transcript:
        await update.message.reply_text(
            "⚠️ *Error*: No transcript loaded\\. Please send a YouTube URL first\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "🤔 _Analyzing your question\\.\\.\\._", parse_mode=ParseMode.MARKDOWN_V2
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant answering questions about a video transcript. Base your answers strictly on the transcript content. Format your response using markdown.",
                },
                {
                    "role": "user",
                    "content": f"Here's a transcript:\n{transcript}\n\nQuestion: {question}",
                },
            ],
        )

        answer = response.choices[0].message.content
        formatted_answer = f"*Answer:*\n\n{escape_markdown(answer)}"
        await update.message.reply_text(
            formatted_answer, parse_mode=ParseMode.MARKDOWN_V2
        )
        return WAITING_FOR_QUESTION

    except Exception as e:
        error_message = f"⚠️ *Error:* {escape_markdown(str(e))}"
        await update.message.reply_text(error_message, parse_mode=ParseMode.MARKDOWN_V2)
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


def main():
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

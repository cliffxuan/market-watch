from os import getenv

from loguru import logger
from openai import OpenAI

# Replace with your tokens
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
    ALLOWED_USER_IDS,
    GPT_MODEL,
    OPENAI_API_KEY,
    TELEGRAM_BOT_TOKEN,
    YOUTUBE_API_KEY,
    YOUTUBE_TRANSCRIPT_API_PROXY,
)
from market_watch.youtube import Video, get_video_id, get_video_metadata

# Conversation states
WAITING_FOR_QUESTION = 1

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    if ALLOWED_USER_IDS and update.effective_user.id not in ALLOWED_USER_IDS:
        message = "*Sorry, you are not authorized to use this bot\.*"
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)
        return

    welcome_message = (
        "*Welcome to YouTube Transcript Analyzer\!*\n\n"
        "Send me a YouTube URL and I'll provide:\n"
        "• A concise summary of the content\n"
        "• Ability to ask questions about the video\n\n"
        "_Just paste a YouTube URL to begin\!_"
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN_V2)


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
        return f"Error fetching transcript: {e}"


async def handle_youtube_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle YouTube URLs sent to the bot."""
    if ALLOWED_USER_IDS and update.effective_user.id not in ALLOWED_USER_IDS:
        message = "*Sorry, you are not authorized to use this bot\.*"
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)
        return ConversationHandler.END

    url = update.message.text
    logger.info(f"start handle_youtube_url url={url}")
    video_id = get_video_id(url)
    context.user_data["url"] = url

    if not video_id:
        await update.message.reply_text("⚠️ Please send a valid YouTube URL.")
        return ConversationHandler.END

    await update.message.reply_text(
        "🔄 Processing video content\nFetching transcript and generating summary..."
    )

    metadata = get_video_metadata(video_id, YOUTUBE_API_KEY)
    video = Video.process_video(metadata)
    context.user_data["transcript"] = "\n".join(
        [item for item in [video.title, video.description, video.captions] if item]
    )

    # Send without markdown parsing to preserve simple formatting
    await update.message.reply_text(video.summary)

    await update.message.reply_text(
        "💡 Ask Questions\n"
        "You can now ask specific questions about the content.\n"
        "Send /done when you're finished."
    )
    return WAITING_FOR_QUESTION


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user questions about the transcript."""
    question = update.message.text
    logger.info(f'start handle_question "{question}"')

    if get_video_id(question):
        await update.message.reply_text(
            f"⚠️ New video {question} sent before finishing the last one."
            "Please end the last session with /done first."
        )
        return WAITING_FOR_QUESTION

    transcript = context.user_data.get("transcript", "")

    if not transcript:
        await update.message.reply_text(
            "⚠️ Error: No transcript loaded. Please send a YouTube URL first."
        )
        return ConversationHandler.END

    await update.message.reply_text("🤔 Analyzing your question...")

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
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
        await update.message.reply_text(f"⚠️ Error: {e}")
        return WAITING_FOR_QUESTION


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """End the conversation."""
    await update.message.reply_text(
        "✅ *Session ended*\n"
        "_Send another YouTube URL whenever you want to analyze a new video\\!_",
        parse_mode=ParseMode.MARKDOWN_V2,
    )
    context.user_data.clear()
    logger.info(f"finish conversation url={context.user_data.get('url')}")
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

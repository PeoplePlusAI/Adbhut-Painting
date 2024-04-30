import openai
from dotenv import load_dotenv
import os

load_dotenv(
    dotenv_path="ops/.env"
)

openai.api_key = os.getenv("OPENAI_API_KEY")


def text_to_speech(text):
    """Convert text to speech using Whisper's TTS."""
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        response_format="wav",
        input=text
    )
    return response
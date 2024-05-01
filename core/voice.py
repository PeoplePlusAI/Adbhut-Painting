import openai
import requests
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
import base64
from dotenv import load_dotenv
import os
import io

load_dotenv(
    dotenv_path="ops/.env"
)

openai.api_key = os.getenv("OPENAI_API_KEY")

DUBVERSE_API_KEY = os.getenv("DUBVERSE_API_KEY")


def dubverse_tts(text, speaker_no=9, api_key=DUBVERSE_API_KEY):
    base_url = "https://macaque.dubverse.ai/api/merlin/services/tts"
    endpoint = "/text-to-speech"
    headers = {
        "X-API-Key": api_key
    }
    payload = {
        "text": text,
        "speaker_no": speaker_no
    }

    try:
        response = requests.post(base_url + endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            return response.content
        else:
            print("Error:", response.text)
            return None
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None


def text_to_speech(text):
    """Convert text to speech using Whisper's TTS."""
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        response_format="wav",
        input=text
    )
    return response
    
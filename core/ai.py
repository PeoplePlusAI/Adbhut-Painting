from dotenv import load_dotenv
from utils.file import convert_to_base64
from core.face_detection import detect_people_yolo
from core.voice import text_to_speech, dubverse_tts
from llms.gpt4vision import get_openai_response
from llms.llava import get_llava_response
import base64
import random
import requests
import time
import os

load_dotenv(dotenv_path="ops/.env")

LLAVA_URL = os.getenv("LLAVA_URL")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

previous_response = ""

with open("prompts/main.txt", "r") as f:
    prompt = f.read()

def get_ai_response(encoded_img):
    global previous_response
    prompt = prompt.format(previous_response)
    response = get_openai_response(OPENAI_API_KEY, prompt, encoded_img)
    if not response:
      response = get_llava_response(encoded_img, prompt)
    else:
      print(response)
    previous_response = response
    return response

def respond_voice(contents, person_detected=False):
    person_detected = detect_people_yolo(contents)
    if person_detected:
        encoded_img = convert_to_base64(contents)
        response_text = get_ai_response(encoded_img)
    else:
        response_text = ""
    if response_text:
        audio_response = dubverse_tts(response_text)
        encoded_audio = base64.b64encode(audio_response.content).decode("utf-8")
        return {"response": response_text, "audio": encoded_audio, "expression": "talking"}
    else:
        expression = random.choice(["deadface", "surprised", "idle"])
        return {"response": "", "audio": "", "expression": expression}

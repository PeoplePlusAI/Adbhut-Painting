from dotenv import load_dotenv
from utils.file import convert_to_base64
from core.face_detection import detect_people_yolo
from core.voice import text_to_speech, dubverse_tts
import base64
import random
import requests
import time
import os

load_dotenv(dotenv_path="ops/.env")

LLAVA_URL = os.getenv("LLAVA_URL")

with open("prompts/main.txt", "r") as f:
    prompt = f.read()


def compose_body(encoded_img, prompt=prompt, model="llava"):
    return {
        "model": model,
        "prompt": prompt,
        "images": [encoded_img],
        "stream": False
    }


def llava_chat(encoded_img):
    body = compose_body(encoded_img)
    start = time.time()
    response = requests.post(LLAVA_URL, json=body)
    end = time.time()
    print(end - start)
    if response.status_code != 200:
        print(f"Something went wrong. Status code is {response.status_code}")
        return ""
    print(response.json())
    response = response.json().get("response", "").replace("</s>", "").strip()
    return response


def respond_voice(contents, person_detected=False):
    person_detected = detect_people_yolo(contents)
    if person_detected:
        encoded_img = convert_to_base64(contents)
        response_text = llava_chat(encoded_img)
    else:
        response_text = ""
    if response_text:
        audio_response = dubverse_tts(response_text)
        encoded_audio = base64.b64encode(audio_response.content).decode("utf-8")
        return {"response": response_text, "audio": encoded_audio, "expression": "talking"}
    else:
        expression = random.choice(["deadface", "surprised", "idle"])
        return {"response": "", "audio": "", "expression": expression}

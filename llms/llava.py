from dotenv import load_dotenv
from utils.ollama_utils import compose_body
import requests
import os

load_dotenv(dotenv_path="ops/.env")

LLAVA_URL = os.getenv("LLAVA_URL")


def get_llava_response(encoded_img, prompt, model="llava"):
    body = compose_body(encoded_img, prompt)
    response = requests.post(LLAVA_URL, json=body)
    print(response.json())
    if response.status_code != 200:
        print(f"Something went wrong. Status code is {response.status_code}")
        return ""
    response = response.json().get("response", "").replace("</s>", "").strip()
    return response
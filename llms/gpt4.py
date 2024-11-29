from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="ops/.env")

api_key = os.getenv("OPENAI_API_KEY")


def get_openai_response(prompt):
    client = OpenAI(
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content
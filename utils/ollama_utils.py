def compose_body(encoded_img, prompt, model):
    return {
        "model": model,
        "prompt": prompt,
        "images": [encoded_img],
        "stream": False
    }
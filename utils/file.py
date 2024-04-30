import base64

def convert_to_base64(contents):
    # Convert bytes to base64 string
    base64_image = base64.b64encode(contents).decode()
    return base64_image
from deepface import DeepFace
from PIL import Image
import numpy as np
import io

def load_image(image):
    img = Image.open(io.BytesIO(image))
    img = np.array(img)
    return img

def verify_face(img1, img2):
    img2 = load_image(img2)
    result = DeepFace.verify(img1, img2)
    return result["verified"]
import cv2
import time
import redis
import numpy as np

redis_client = redis.Redis(host='localhost', port=6379, db=0)

net = cv2.dnn.readNet("coco_model/yolov3.weights", "coco_model/yolov3.cfg")
classes = []
with open("coco_model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

if output_layers_indices.ndim > 1:  # Checking if the output is a nested list
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]
else:
    output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# Initialize time of last detection and debounce period
last_detection_time = 0
debounce_period = 100  # Adjust the debounce period as needed (in seconds)

def detect_people_yolo(image_bytes):
    global redis_client

    # Retrieve previous count from Redis
    previous_count = int(redis_client.get("previous_count") or 0)

    # Convert bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Ensure the frame was correctly decoded
    if frame is None:
        raise ValueError("Could not decode the image")

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize a set to track detected person IDs
    detected_person_ids = set()

    # Process detections from each of the output layers
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Check if the detected class is 'person' (class_id == 0 for 'person' in COCO dataset)
                detected_person_ids.add(class_id)  # Add the detected person ID to the set

    # Calculate the current count of detected persons
    current_count = len(detected_person_ids)

    # Update previous count in Redis
    redis_client.set("previous_count", current_count)

    # Check if a new person has been detected
    if current_count > previous_count:
        return True  # New person detected
    else:
        return False  # No new person detected
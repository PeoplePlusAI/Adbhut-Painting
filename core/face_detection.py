import cv2
import numpy as np

net = cv2.dnn.readNet("coco-model/yolov3.weights", "coco-model/yolov3.cfg")
classes = []
with open("coco-model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

if output_layers_indices.ndim > 1:  # Checking if the output is a nested list
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]
else:
    output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

def detect_people_yolo(image_bytes):
    # Convert bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Ensure the frame was correctly decoded
    if frame is None:
        raise ValueError("Could not decode the image")

    # Static variable to hold previous detection state
    if not hasattr(detect_people_yolo, "last_count"):
        detect_people_yolo.last_count = 0  # Initialize with no people initially detected

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    current_count = 0  # Count current frame detections

    # Process detections from each of the output layers
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Check if the detected class is 'person' (class_id == 0 for 'person' in COCO dataset)
                if class_id == 0:
                    current_count += 1

    # Check if there is an increase in the number of people detected
    is_new_person = current_count > detect_people_yolo.last_count
    detect_people_yolo.last_count = current_count  # Update the count
    return is_new_person
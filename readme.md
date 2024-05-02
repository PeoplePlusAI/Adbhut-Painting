# People+AI Talking Painting

This is the code for the AI painting that talks to you. The following AI generated painting is framed in our office. 

![AI Painting](/docs/ai-painting.png)

The plan is to put a tablet behind the painting and have it talks when it detects a person infront of it, yes! like the paintings in harrypotter movies.


## How to run the code

1. Clone the repo
2. Create a `ops/.env` file and add the following variables
```bash
OPENAI_API_KEY=your_openai_api_key
LLAVA_URL=ollama_url
```

3. Install the dependencies
```bash
pip install -r requirements.txt
```

4. Run the following commands to get the yolov3 weights
```bash
wget https://pjreddie.com/media/files/yolov3.weights -O coco_model/yolov3.weights
```

5. Run the web server

```bash
uvicorn app:app --reload
```

6. Open the browser and go to `http://localhost:8000/`

7. Move in and move out of the camera to see the painting talks to you.


## What works now

* We use YOLO to detect people infront of the camera
* LLAVA model for making a comment about the person based on the prompt
* OpenAI TTS and other TTS models to make the painting talk
* The frontend code will keep capturing the video frames and send it to the AI API for processing.
* Once the AI API return an audio, the frontend code will pause sending the video frames and play the audio.


## What needs to be done

* Modify the frontend code to replace the video feed with robot face animation based on the requirements in this [documemt](https://docs.google.com/document/d/1caHAHjxJxG-dlpAiR1raLPVOrsJIDzLPOyE027YJxO0/edit?usp=sharing)

* Modify the prompt and test to find the right one that works for the painting.




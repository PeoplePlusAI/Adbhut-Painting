from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from core.face_detection import detect_people_yolo
from utils.file import convert_to_base64
from core.ai import respond, respond_voice


app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (you can specify specific origins)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow GET, POST, and OPTIONS methods
    allow_headers=["*"],  # Allow all headers
)


@app.post("/respond/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    person_detected = detect_people_yolo(contents)

    if person_detected:
        encoded_img = convert_to_base64(contents)
        return JSONResponse(content={"response": respond(encoded_img)})
    else:
        return JSONResponse(content={})

    
@app.post("/respond_voice/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    person_detected = detect_people_yolo(contents)

    if person_detected:
        encoded_img = convert_to_base64(contents)
        response = respond_voice(encoded_img)

        return JSONResponse(content=response)
    else:
        return JSONResponse(content={})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the index.html file."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


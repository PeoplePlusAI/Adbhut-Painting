from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from utils.redis_utils import set_previous_count
import os
from core.ai import respond_voice, respond_farewell

load_dotenv(dotenv_path="ops/.env")

from core.models import VoiceResponseModel
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (you can specify specific origins)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow GET, POST, and OPTIONS methods
    allow_headers=["*"],  # Allow all headers
)


@app.on_event("startup")
def startup():
    set_previous_count()
    
@app.post("/respond_voice/")
async def detect(file: UploadFile = File(...)) -> VoiceResponseModel:
    contents = await file.read()
    response = respond_farewell(contents)
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the index.html file."""
    STATIC_FILE_URL = os.getenv("STATIC_FILE_URL", "http://127.0.0.1:8000/static") 
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/respond_voice/")
    return templates.TemplateResponse("index.html", {"request": request,"static_file_url": STATIC_FILE_URL, "api_url": API_URL})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


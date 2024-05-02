from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from core.ai import respond_voice

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

    
@app.post("/respond_voice/")
async def detect(file: UploadFile = File(...)) -> VoiceResponseModel:
    contents = await file.read()
    response = respond_voice(contents)
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the index.html file."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


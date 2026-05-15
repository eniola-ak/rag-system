from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from src.ingest import ingest_pdf, ingest_url, ingest_youtube
from src.chat import chat

app = FastAPI(title="Multi-Source RAG System")

# ---- Health Check ----
@app.get("/")
def health():
    return {"status": "running", "project": "Multi-Source RAG System"}

# ---- Ingest: PDF ----
@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"data/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = ingest_pdf(temp_path)
    return result

# ---- Ingest: URL ----
class URLPayload(BaseModel):
    url: str

@app.post("/ingest/url")
def ingest_url_endpoint(payload: URLPayload):
    result = ingest_url(payload.url)
    return result

# ---- Ingest: YouTube ----
class YouTubePayload(BaseModel):
    video_url: str

@app.post("/ingest/youtube")
def ingest_youtube_endpoint(payload: YouTubePayload):
    result = ingest_youtube(payload.video_url)
    return result

# ---- Chat ----
class ChatPayload(BaseModel):
    question: str
    top_k: int = 3

@app.post("/chat")
def chat_endpoint(payload: ChatPayload):
    result = chat(payload.question, payload.top_k)
    return result

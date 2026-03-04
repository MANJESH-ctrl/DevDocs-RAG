import os
import json
import shutil
from uuid import uuid4
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import UPLOAD_DIR, STATIC_DIR
from ingestion import ingest_document, get_embeddings, get_index
from query import get_reranker, stream_rag_response

app = FastAPI(title="DEV-DOCS RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

sessions: Dict[str, Dict[str, Any]] = {}


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[HistoryMessage] = []


@app.on_event("startup")
def warmup():
    print("🔥 Warming up models...")
    get_embeddings()
    get_reranker()
    get_index()
    print("✅ All models loaded and ready.")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = os.path.join(STATIC_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_file(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    session_id = str(uuid4())
    file_path  = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    sessions[session_id] = {
        "status":    "processing",
        "stage":     "Starting...",
        "file_name": file.filename,
        "error":     None,
    }

    background_tasks.add_task(run_ingestion, file_path, session_id)
    return {"session_id": session_id, "status": "processing"}


def run_ingestion(file_path: str, session_id: str):
    def update_stage(stage: str):
        sessions[session_id]["stage"] = stage

    try:
        ingest_document(file_path, session_id, progress_cb=update_stage)
        sessions[session_id]["status"] = "ready"
        sessions[session_id]["stage"]  = "Done!"
    except Exception as e:
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["error"]  = str(e)


@app.get("/status/{session_id}")
def get_status(session_id: str):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found.")
    return {
        "status":    s["status"],
        "stage":     s.get("stage", ""),
        "error":     s.get("error"),
        "file_name": s.get("file_name"),
    }


@app.post("/chat/{session_id}")
async def chat(session_id: str, req: ChatRequest):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found. Server may have restarted — please re-upload.")
    if s["status"] != "ready":
        raise HTTPException(400, f"Document not ready. Current status: {s['status']}")

    history = [{"role": m.role, "content": m.content} for m in req.history[-6:]]

    async def generate():
        try:
            async for event in stream_rag_response(req.question, session_id, history):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

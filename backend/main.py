import os
import shutil
from uuid import uuid4
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import UPLOAD_DIR
from ingestion_utils import ingest_document
from query_utils import get_rag_chain, hybrid_retriever

app = FastAPI(title="DEV-DOCS RAG")

sessions: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    question: str


def get_session_or_404(session_id: str) -> Dict[str, Any]:
    s = sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found.")
    return s


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_file(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    session_id = str(uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    sessions[session_id] = {
        "status": "processing",
        "file_name": file.filename,
    }

    background_tasks.add_task(run_ingestion, file_path, session_id)

    return {"session_id": session_id, "status": "processing"}


def run_ingestion(file_path: str, session_id: str):
    try:
        ingest_document(file_path, session_id)
        sessions[session_id]["status"] = "ready"
    except Exception as e:
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["error"] = str(e)


@app.get("/status/{session_id}")
def get_status(session_id: str):
    s = get_session_or_404(session_id)
    return {
        "status": s["status"],
        "error": s.get("error"),
        "file_name": s.get("file_name"),
    }


@app.post("/chat/{session_id}")
def chat(session_id: str, req: ChatRequest):
    s = get_session_or_404(session_id)
    if s["status"] != "ready":
        raise HTTPException(400, "Document not ready yet.")

    chain = get_rag_chain(session_id)
    answer = chain.invoke(req.question)

    docs = hybrid_retriever(req.question, session_id)
    sources = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        headers = " > ".join(v for k, v in doc.metadata.items() if k.startswith("Header"))
        sources.append(
            {
                "source": source,
                "section": headers,
                "preview": doc.page_content[:150],
            }
        )

    return {"answer": answer, "sources": sources}


@app.post("/chat/{session_id}/stream")
def chat_stream(session_id: str, req: ChatRequest):
    s = get_session_or_404(session_id)
    if s["status"] != "ready":
        raise HTTPException(400, "Document not ready yet.")

    chain = get_rag_chain(session_id)
    docs = hybrid_retriever(req.question, session_id)
    sources = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        headers = " > ".join(v for k, v in doc.metadata.items() if k.startswith("Header"))        
        sources.append(
            {
                "source": source,
                "section": headers,
                "preview": doc.page_content[:150],
            }
        )

    def generate():
        import json
        yield f"__SOURCES__{json.dumps(sources)}__SOURCES__\n"
        for chunk in chain.stream(req.question):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")

"""
app.py
~~~~~~
FastAPI backend for the AI Customer Support RAG system.

Endpoints:
  POST /upload  – Upload a document (PDF / TXT / MD)
  POST /ask     – Ask a question
  GET  /status  – Health check + document count
  DELETE /reset – Clear the vector store
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import RAGPipeline
from upload_docs import SUPPORTED_EXTENSIONS, process_bytes

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Customer Support Assistant",
    description="RAG-powered customer support API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    source: str
    confidence: float


class StatusResponse(BaseModel):
    status: str
    document_chunks: int


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"message": "AI Customer Support Assistant is running 🚀"}


@app.get("/status", response_model=StatusResponse, tags=["Health"])
def status():
    return StatusResponse(status="ok", document_chunks=pipeline.document_count())


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        texts, metadatas = process_bytes(contents, file.filename)
        pipeline.index_documents(texts, metadatas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    return UploadResponse(
        filename=file.filename,
        chunks_added=len(texts),
        total_chunks=pipeline.document_count(),
        message=f"'{file.filename}' processed and indexed successfully.",
    )


@app.post("/ask", response_model=AskResponse, tags=["Query"])
def ask_question(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = pipeline.query(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")

    return AskResponse(
        answer=result["answer"],
        source=result["source"],
        confidence=result["confidence"],
    )


@app.delete("/reset", tags=["Documents"])
def reset_vector_store():
    pipeline.clear()
    return {"message": "Vector store cleared successfully."}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

"""
upload_docs.py
~~~~~~~~~~~~~~
Loads PDF / TXT / Markdown files, splits them into chunks,
and returns (texts, metadatas) ready for the vector store.
No LangChain dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


# ---------------------------------------------------------------------------
# Text loaders
# ---------------------------------------------------------------------------

def _load_pdf(file_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(file_path)
    if ext in {".txt", ".md", ".markdown"}:
        return _load_text(file_path)
    raise ValueError(f"Unsupported file type: '{ext}'")


# ---------------------------------------------------------------------------
# Simple recursive text splitter
# ---------------------------------------------------------------------------

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by character count."""
    separators = ["\n\n", "\n", ". ", " "]

    def _split(t: str, seps: List[str]) -> List[str]:
        if len(t) <= chunk_size:
            return [t] if t.strip() else []

        sep = seps[0] if seps else ""
        remaining_seps = seps[1:] if len(seps) > 1 else []

        parts = t.split(sep) if sep else list(t)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size and remaining_seps:
                    chunks.extend(_split(part, remaining_seps))
                else:
                    current = part
                    continue
                current = ""

        if current.strip():
            chunks.append(current)

        return chunks

    raw_chunks = _split(text, separators)

    # Add overlap
    if overlap > 0 and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev_tail = raw_chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + raw_chunks[i])
        return overlapped

    return raw_chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_file(file_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Load, split, and return (texts, metadatas).
    """
    source = Path(file_path).name
    raw_text = load_file(file_path)
    chunks = split_text(raw_text)

    texts = chunks
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    print(f"[upload_docs] '{source}' → {len(chunks)} chunks ✔")
    return texts, metadatas


def process_bytes(file_bytes: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Process raw bytes (from an upload endpoint)."""
    import tempfile

    ext = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        texts, metadatas = process_file(tmp_path)
        for m in metadatas:
            m["source"] = filename
        return texts, metadatas
    finally:
        os.unlink(tmp_path)

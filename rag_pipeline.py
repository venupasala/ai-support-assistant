"""
rag_pipeline.py
~~~~~~~~~~~~~~~
Core RAG pipeline:
  1. Embed the user query
  2. Search FAISS for relevant chunks
  3. Build a prompt with context
  4. Call LLM (OpenAI GPT or a local extractive fallback)
  5. Return answer + source + confidence
No LangChain dependency.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from embedding_model import EmbeddingModel
from vector_store import VectorStore

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()
TOP_K = 4


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant. "
                    "Answer questions using ONLY the provided context. "
                    "If the answer is not in the context, say so clearly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _call_local(context_chunks: List[str]) -> str:
    """
    Simple extractive 'LLM' — picks the most relevant sentences.
    No API key needed.
    """
    if not context_chunks:
        return (
            "I'm sorry, I couldn't find relevant information in the uploaded documents. "
            "Please upload relevant documents first."
        )
    answer = context_chunks[0].strip()
    sentences = answer.split(". ")
    short = ". ".join(sentences[:4])
    if not short.endswith("."):
        short += "."
    return short


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str, chunks: List[str]) -> str:
    context = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks))
    return textwrap.dedent(f"""\
        You are a customer support assistant.
        Use ONLY the context below to answer the question.
        If the answer is not in the context, say "I don't have enough information."

        Context:
        {context}

        Question: {question}

        Answer:""")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(self) -> None:
        self._embedding_model = EmbeddingModel()
        self._vector_store = VectorStore(self._embedding_model)

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self._vector_store.add_documents(texts, metadatas)

    def document_count(self) -> int:
        return self._vector_store.document_count()

    def clear(self) -> None:
        self._vector_store.clear()

    # ── Querying ──────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        results: List[Tuple[str, Dict[str, Any], float]] = (
            self._vector_store.search(question, k=TOP_K)
        )

        if not results:
            return {
                "answer": "No relevant documents found. Please upload support documents first.",
                "source": "N/A",
                "confidence": 0.0,
                "chunks": [],
            }

        chunks = [text for text, _, _ in results]
        scores = [score for _, _, score in results]
        sources = [meta.get("source", "unknown") for _, meta, _ in results]

        best_idx = scores.index(max(scores))
        best_source = sources[best_idx]
        confidence = round(float(max(scores)), 4)

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        prompt = _build_prompt(question, chunks)
        if LLM_BACKEND == "openai":
            answer = _call_openai(prompt)
        else:
            answer = _call_local(chunks)

        return {
            "answer": answer,
            "source": best_source,
            "confidence": confidence,
            "chunks": chunks,
        }

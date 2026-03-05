"""
vector_store.py
~~~~~~~~~~~~~~~
Pure-numpy vector store with cosine similarity search.
No FAISS, no ChromaDB — fully compatible with Python 3.14.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv

from embedding_model import EmbeddingModel

load_dotenv()

PERSIST_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_db")


class VectorStore:
    """Persistent vector store using numpy arrays + JSON metadata."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._emb = embedding_model
        self._dim = embedding_model.dimension
        self._persist_dir = PERSIST_DIR

        # In-memory storage
        self._vectors: np.ndarray | None = None  # shape (N, dim)
        self._metadata: List[Dict[str, Any]] = []
        self._texts: List[str] = []

        self._load()
        n = 0 if self._vectors is None else self._vectors.shape[0]
        print(f"[VectorStore] Loaded ({n} vectors, dim={self._dim}) ✔")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _vec_path(self) -> str:
        return os.path.join(self._persist_dir, "vectors.npy")

    def _meta_path(self) -> str:
        return os.path.join(self._persist_dir, "metadata.json")

    def _save(self) -> None:
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        if self._vectors is not None and self._vectors.shape[0] > 0:
            np.save(self._vec_path(), self._vectors)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump({"texts": self._texts, "metadata": self._metadata}, f)

    def _load(self) -> None:
        vec_path = self._vec_path()
        meta_path = self._meta_path()
        if os.path.exists(vec_path) and os.path.exists(meta_path):
            self._vectors = np.load(vec_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._texts = data.get("texts", [])
            self._metadata = data.get("metadata", [])

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not texts:
            return
        new_vecs = self._emb.embed(texts)  # (N, dim) float32
        if self._vectors is None or self._vectors.shape[0] == 0:
            self._vectors = new_vecs
        else:
            self._vectors = np.vstack([self._vectors, new_vecs])
        self._texts.extend(texts)
        self._metadata.extend(metadatas)
        self._save()
        print(f"[VectorStore] Added {len(texts)} chunks ✔")

    def clear(self) -> None:
        self._vectors = None
        self._texts = []
        self._metadata = []
        self._save()
        print("[VectorStore] Cleared ✔")

    # ------------------------------------------------------------------
    # Search  (cosine similarity on normalised vectors)
    # ------------------------------------------------------------------

    def search(
        self, query: str, k: int = 4
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []

        query_vec = self._emb.embed_query(query)  # (dim,)
        # Cosine similarity = dot product for L2-normalised vectors
        scores = self._vectors @ query_vec  # (N,)
        k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((
                self._texts[idx],
                self._metadata[idx],
                float(scores[idx]),
            ))
        return results

    def document_count(self) -> int:
        if self._vectors is None:
            return 0
        return self._vectors.shape[0]

"""
embedding_model.py
~~~~~~~~~~~~~~~~~~
Wraps sentence-transformers for generating text embeddings.
No LangChain dependency — works directly with Python 3.14.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_MODEL = os.getenv(
    "HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class EmbeddingModel:
    """Generates text embeddings using sentence-transformers."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(HUGGINGFACE_MODEL)
        self._dim = self._model.get_sentence_embedding_dimension()
        print(f"[EmbeddingModel] Loaded '{HUGGINGFACE_MODEL}' (dim={self._dim}) ✔")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, dim) float32 array."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns (dim,) float32 array."""
        return self.embed([text])[0]

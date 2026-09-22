from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, dependency-light embedder for local demos and CI."""

    name = "hashing-v1"

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for token in features:
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                vectors[row, value % self.dimension] += 1.0 if value & 1 else -1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def create_embedder(model: str | None = None) -> Embedder:
    model = model or os.getenv("SEARCH_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "hashing"))
    return HashingEmbedder() if model == "hashing" else SentenceTransformerEmbedder(model)

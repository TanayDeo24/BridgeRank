from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import numpy as np

from .embeddings import Embedder
from .models import Document, SearchResult
from .text import tokenize


class BM25Index:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents, self.k1, self.b = documents, k1, b
        self.tokens = [tokenize(d.searchable_text) for d in documents]
        self.tf = [Counter(row) for row in self.tokens]
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        df = Counter(t for row in self.tokens for t in set(row))
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        values = np.zeros(len(self.documents), dtype=np.float32)
        for i, (freq, row) in enumerate(zip(self.tf, self.tokens)):
            dl = len(row)
            for token in tokenize(query):
                tf = freq[token]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                values[i] += self.idf.get(token, 0) * tf * (self.k1 + 1) / max(denom, 1e-9)
        return values


class DenseIndex:
    def __init__(
        self,
        documents: list[Document],
        embedder: Embedder,
        index_type: str = "flat",
        vectors: np.ndarray | None = None,
    ):
        self.documents, self.embedder, self.index_type = documents, embedder, index_type
        self.vectors = (
            vectors
            if vectors is not None
            else embedder.encode([d.searchable_text for d in documents])
        )
        self._faiss = None
        try:
            import faiss

            if index_type == "hnsw":
                self._faiss = faiss.IndexHNSWFlat(embedder.dimension, 32)
                self._faiss.hnsw.efConstruction = 80
                self._faiss.hnsw.efSearch = 64
            else:
                self._faiss = faiss.IndexFlatIP(embedder.dimension)
            self._faiss.add(self.vectors)
        except ImportError:
            pass

    def scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query])[0]
        if self._faiss is None:
            return self.vectors @ q
        distances, indices = self._faiss.search(q.reshape(1, -1), len(self.documents))
        scores = np.full(len(self.documents), -1.0, dtype=np.float32)
        for score, index in zip(distances[0], indices[0]):
            if index >= 0:
                scores[index] = score
        return scores


def _normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0 or float(values.max() - values.min()) < 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())


class HybridSearchEngine:
    def __init__(
        self,
        documents: list[Document],
        embedder: Embedder,
        alpha: float = 0.55,
        index_type: str = "flat",
        vectors: np.ndarray | None = None,
    ):
        self.documents, self.alpha = documents, alpha
        self.bm25 = BM25Index(documents)
        self.dense = DenseIndex(documents, embedder, index_type, vectors)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        lexical, semantic = self.bm25.scores(query), self.dense.scores(query)
        if mode == "bm25":
            combined = _normalize(lexical)
        elif mode == "dense":
            combined = _normalize(semantic)
        elif mode == "hybrid":
            combined = (1 - self.alpha) * _normalize(lexical) + self.alpha * _normalize(semantic)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        allowed = np.ones(len(self.documents), dtype=bool)
        for key, value in (filters or {}).items():
            allowed &= np.array(
                [str(getattr(d, key, d.metadata.get(key))) == str(value) for d in self.documents]
            )
        ranked = [i for i in np.argsort(-combined) if allowed[i]][:top_k]
        return [
            SearchResult(
                document=self.documents[i],
                score=float(combined[i]),
                rank=rank,
                retrieval_scores={"bm25": float(lexical[i]), "dense": float(semantic[i])},
            )
            for rank, i in enumerate(ranked, 1)
        ]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        scores = self.model.predict([(query, r.document.searchable_text) for r in results])
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [
            replace(results[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, 1)
        ]

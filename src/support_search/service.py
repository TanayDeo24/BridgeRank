from __future__ import annotations

import hashlib
import time
import uuid

from .artifacts import load_artifact
from .config import Settings
from .embeddings import create_embedder
from .io import load_documents
from .retrieval import CrossEncoderReranker, HybridSearchEngine
from .storage import EventStore


class SearchService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = create_embedder(settings.embedding_model)
        self.manifest: dict = {}
        vectors = None
        if (settings.artifact_dir / "manifest.json").exists():
            self.documents, vectors, self.manifest = load_artifact(
                settings.artifact_dir, self.embedder.name
            )
        elif settings.environment == "production":
            raise RuntimeError("A validated index artifact is required in production")
        else:
            self.documents = load_documents(settings.data_path)
        self.engine = HybridSearchEngine(
            self.documents, self.embedder, settings.hybrid_alpha, settings.index_type, vectors
        )
        self.store = EventStore(settings.database_url or settings.database_path)
        self.reranker = (
            CrossEncoderReranker(settings.reranker_model) if settings.reranker_model else None
        )

    def search(
        self, query: str, top_k: int, filters: dict[str, str], mode: str, rerank: bool = False
    ):
        started = time.perf_counter()
        candidate_k = max(top_k, min(self.settings.candidate_k, len(self.documents)))
        results = self.engine.search(query, candidate_k if rerank else top_k, filters, mode)
        if rerank:
            if self.reranker is None:
                raise ValueError("Reranking is not configured")
            results = self.reranker.rerank(query, results, top_k)
        latency = (time.perf_counter() - started) * 1000
        query_id = str(uuid.uuid4())
        self.store.record_query(
            {
                "query_id": query_id,
                "query_hash": hashlib.sha256(query.strip().lower().encode()).hexdigest(),
                "query": query,
                "mode": mode,
                "latency_ms": round(latency, 3),
                "result_ids": [r.document.id for r in results],
                "failed": not bool(results),
                "metadata": {
                    "filters": filters,
                    "embedding_model": self.embedder.name,
                    "reranked": rerank,
                    "artifact_source_sha256": self.manifest.get("source_sha256"),
                },
            },
            retain_text=self.settings.retain_query_text,
        )
        return query_id, results, latency

    def feedback(self, query_id: str, document_id: str, relevance: int) -> str:
        feedback_id = str(uuid.uuid4())
        self.store.record_feedback(feedback_id, query_id, document_id, relevance)
        return feedback_id

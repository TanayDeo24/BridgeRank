"""Configurable, reproducible ranking experiments (Sections 14-16).

An ``ExperimentConfig`` fully determines one ranking pipeline: which
features are active and at what weight, which fusion strategy combines
them, how deep the candidate pool and reranking stage go, which models are
used, and which data split it is evaluated against. Running the same config
twice (same commit, same data) reproduces the same numbers exactly — there
is no unseeded randomness anywhere in the scoring path (the one seeded
random component, the learned-linear model's gradient descent, records its
seed in the config and in the saved model file).

Stage semantics (Section 15) are preserved explicitly:

    first-stage retrieval  -> BM25Index / DenseIndex raw scores (always both
                               computed; which ones *count* is controlled by
                               retrieval_features weights, e.g. an alpha=0
                               weight excludes dense without skipping its
                               computation, keeping cost measurement honest)
    candidate pool          -> top `candidate_k` by the fused retrieval-stage
                               score
    second-stage reranking  -> optional cross-encoder over the top
                               `rerank_depth` (<= candidate_k) candidates,
                               fused with the retrieval-stage score via
                               `rerank_weights`
    final result depth      -> top `top_k` after any reranking
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embeddings import create_embedder
from .evaluation import ndcg_at_k, percentile, precision_at_k, recall_at_k, reciprocal_rank
from .features import FEATURE_REGISTRY, FeatureContext
from .fusion import FUSION_STRATEGIES, normalize
from .io import load_documents, load_queries
from .models import Document
from .ranking_model import LogisticRankModel
from .retrieval import BM25Index, CrossEncoderReranker, DenseIndex

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class ExperimentConfig:
    id: str
    description: str = ""
    embedding_model: str = "hashing"
    index_type: str = "flat"
    candidate_k: int = 50
    rerank_depth: int = 0  # 0 disables reranking
    top_k: int = 5
    retrieval_features: dict[str, float] = field(
        default_factory=lambda: {"bm25": 0.45, "dense": 0.55}
    )
    fusion: str = "weighted_sum"
    normalization: str = "minmax"
    reranker_model: str | None = None
    rerank_weights: dict[str, float] = field(
        default_factory=lambda: {"retrieval_score": 0.0, "cross_encoder": 1.0}
    )
    learned_model_path: str | None = None
    seed: int = 42
    split: str = "test"
    git_commit: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "embedding_model": self.embedding_model,
            "index_type": self.index_type,
            "candidate_k": self.candidate_k,
            "rerank_depth": self.rerank_depth,
            "top_k": self.top_k,
            "retrieval_features": self.retrieval_features,
            "fusion": self.fusion,
            "normalization": self.normalization,
            "reranker_model": self.reranker_model,
            "rerank_weights": self.rerank_weights,
            "learned_model_path": self.learned_model_path,
            "seed": self.seed,
            "split": self.split,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> ExperimentConfig:
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class EngineBundle:
    documents: list[Document]
    bm25: BM25Index
    dense: DenseIndex
    reranker: CrossEncoderReranker | None
    learned_model: LogisticRankModel | None


_BUNDLE_CACHE: dict[tuple[str, str, int], EngineBundle] = {}


def build_engine_bundle(
    documents: list[Document],
    config: ExperimentConfig,
    cache: bool = True,
) -> EngineBundle:
    # Hashing document ids (not id(documents)) makes the cache key correct
    # even if two different document sets share the same embedding_model/
    # index_type within one process (e.g. tests) -- a plain object-identity
    # or (model, index_type)-only key would silently return the wrong
    # corpus's index in that case.
    doc_signature = hash(tuple(d.id for d in documents))
    cache_key = (config.embedding_model, config.index_type, doc_signature)
    if cache and cache_key in _BUNDLE_CACHE:
        base = _BUNDLE_CACHE[cache_key]
    else:
        embedder = create_embedder(config.embedding_model)
        bm25 = BM25Index(documents)
        dense = DenseIndex(documents, embedder, config.index_type)
        base = EngineBundle(documents, bm25, dense, reranker=None, learned_model=None)
        if cache:
            _BUNDLE_CACHE[cache_key] = base

    reranker = None
    if config.rerank_depth > 0 and config.reranker_model:
        reranker = CrossEncoderReranker(config.reranker_model)

    learned_model = None
    if config.fusion == "learned_linear" and config.learned_model_path:
        learned_model = LogisticRankModel.load(Path(config.learned_model_path))

    return EngineBundle(base.documents, base.bm25, base.dense, reranker, learned_model)


def _retrieval_feature_matrix(
    query: str,
    documents: list[Document],
    raw_scores: dict[str, np.ndarray],
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    matrix: dict[str, np.ndarray] = {}
    for name in feature_names:
        feature = FEATURE_REGISTRY[name]
        values = np.array(
            [
                feature.extract(
                    FeatureContext(
                        query=query,
                        document=doc,
                        raw_scores={k: float(v[i]) for k, v in raw_scores.items()},
                    )
                )
                for i, doc in enumerate(documents)
            ],
            dtype=np.float32,
        )
        matrix[name] = values
    return matrix


def _fuse(config: ExperimentConfig, feature_matrix: dict[str, np.ndarray], bundle: EngineBundle) -> np.ndarray:
    strategy = FUSION_STRATEGIES[config.fusion]
    if config.fusion == "reciprocal_rank_fusion":
        return strategy(feature_matrix)
    if config.fusion == "learned_linear":
        if bundle.learned_model is None:
            raise ValueError("learned_linear fusion requires learned_model_path in the config")
        return strategy(
            feature_matrix,
            bundle.learned_model.coefficients(),
            bundle.learned_model.bias,
            config.normalization,
        )
    return strategy(feature_matrix, config.retrieval_features, config.normalization)


def score_query(
    config: ExperimentConfig,
    bundle: EngineBundle,
    query: str,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Returns (ranked_doc_ids[:top_k], per-feature raw scores for the top
    result (for inspectability), latency-by-stage in ms)."""
    documents = bundle.documents
    t0 = time.perf_counter()
    bm25_scores = bundle.bm25.scores(query)
    dense_scores = bundle.dense.scores(query)
    raw_scores = {"bm25": bm25_scores, "dense": dense_scores}
    retrieval_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    active_names = [n for n in config.retrieval_features if n in FEATURE_REGISTRY]
    # bm25/dense are always needed as inputs even if weighted 0, plus any static features requested.
    needed = sorted(set(active_names) | {"bm25", "dense"})
    feature_matrix = _retrieval_feature_matrix(query, documents, raw_scores, needed)
    fused = _fuse(config, {k: feature_matrix[k] for k in active_names} or feature_matrix, bundle)
    feature_ms = (time.perf_counter() - t1) * 1000

    order = list(np.argsort(-fused))
    candidate_order = order[: config.candidate_k]

    rerank_ms = 0.0
    if config.rerank_depth > 0 and bundle.reranker is not None:
        t2 = time.perf_counter()
        rerank_pool = candidate_order[: config.rerank_depth]
        pool_docs = [documents[i] for i in rerank_pool]
        ce_scores = bundle.reranker.model.predict([(query, d.searchable_text) for d in pool_docs])
        retrieval_component = normalize(fused[rerank_pool], config.normalization)
        rerank_fused = (
            config.rerank_weights.get("retrieval_score", 0.0) * retrieval_component
            + config.rerank_weights.get("cross_encoder", 1.0)
            * normalize(np.asarray(ce_scores, dtype=np.float32), config.normalization)
        )
        reranked_local_order = np.argsort(-rerank_fused)
        reranked_global = [rerank_pool[i] for i in reranked_local_order]
        final_order = reranked_global + candidate_order[config.rerank_depth :]
        rerank_ms = (time.perf_counter() - t2) * 1000
    else:
        final_order = candidate_order

    top = final_order[: config.top_k]
    doc_ids = [documents[i].id for i in top]
    top_feature_values = (
        {name: float(feature_matrix[name][top[0]]) for name in needed} if top else {}
    )
    latency = {
        "retrieval_ms": retrieval_ms,
        "feature_fusion_ms": feature_ms,
        "rerank_ms": rerank_ms,
        "total_ms": retrieval_ms + feature_ms + rerank_ms,
    }
    return doc_ids, top_feature_values, latency


def load_split_assignment() -> dict[str, str]:
    return json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["assignment"]


def load_corpus() -> list[Document]:
    return load_documents(DATA_DIR / "sample_documents.jsonl")


def load_all_queries() -> list[dict]:
    return load_queries(DATA_DIR / "evaluation_queries.jsonl")


def queries_for_split(split: str) -> list[dict]:
    assignment = load_split_assignment()
    return [q for q in load_all_queries() if assignment.get(q["query"]) == split]


def evaluate_config(
    config: ExperimentConfig,
    documents: list[Document] | None = None,
    queries: list[dict] | None = None,
    primary_k: int = 5,
) -> dict:
    """Runs one experiment config end-to-end and returns a machine-readable
    result dict: relevance metrics, latency percentiles by stage, and the
    exact config/provenance needed to reproduce the run (Section 31)."""
    documents = documents if documents is not None else load_corpus()
    queries = queries if queries is not None else queries_for_split(config.split)
    bundle = build_engine_bundle(documents, config)

    rows = []
    latencies_total: list[float] = []
    latencies_retrieval: list[float] = []
    latencies_feature: list[float] = []
    latencies_rerank: list[float] = []
    error_count = 0
    for item in queries:
        try:
            doc_ids, _, latency = score_query(config, bundle, item["query"])
        except Exception as exc:  # noqa: BLE001 - a failed query is a measured outcome, not a crash
            error_count += 1
            rows.append(
                {
                    "query": item["query"],
                    "precision_at_k": 0.0,
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "ndcg_at_k": 0.0,
                    "failed": True,
                    "error": str(exc),
                }
            )
            continue
        relevance = item["relevance"]
        relevant = {doc_id for doc_id, grade in relevance.items() if grade > 0}
        rows.append(
            {
                "query": item["query"],
                "category": item.get("category", "unknown"),
                "precision_at_k": precision_at_k(doc_ids, relevant, config.top_k),
                "recall_at_k": recall_at_k(doc_ids, relevant, config.top_k),
                "mrr": reciprocal_rank(doc_ids, relevant),
                "ndcg_at_k": ndcg_at_k(doc_ids, relevance, config.top_k),
                "failed": not bool(relevant.intersection(doc_ids)),
                "retrieved": doc_ids,
            }
        )
        latencies_total.append(latency["total_ms"])
        latencies_retrieval.append(latency["retrieval_ms"])
        latencies_feature.append(latency["feature_fusion_ms"])
        latencies_rerank.append(latency["rerank_ms"])

    n = len(rows)
    summary = {
        key: statistics.fmean(r[key] for r in rows) if n else 0.0
        for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k")
    }
    summary.update(
        {
            "failed_query_rate": statistics.fmean(float(r["failed"]) for r in rows) if n else 0.0,
            "error_rate": error_count / n if n else 0.0,
            "query_count": n,
            "p50_latency_ms": percentile(latencies_total, 0.5),
            "p95_latency_ms": percentile(latencies_total, 0.95),
            "p50_retrieval_ms": percentile(latencies_retrieval, 0.5),
            "p95_retrieval_ms": percentile(latencies_retrieval, 0.95),
            "p50_feature_fusion_ms": percentile(latencies_feature, 0.5),
            "p95_feature_fusion_ms": percentile(latencies_feature, 0.95),
            "p50_rerank_ms": percentile(latencies_rerank, 0.5),
            "p95_rerank_ms": percentile(latencies_rerank, 0.95),
        }
    )
    return {
        "config": config.to_dict(),
        "summary": summary,
        "rows": rows,
    }


def log_to_mlflow(result: dict, experiment_name: str = "support-search-ranking-experiments") -> bool:
    """Best-effort MLflow logging (Section 28). Returns True if logged.

    Mirrors the original repository's own try/except ImportError pattern in
    cli.py:run_evaluation (kept, not rebuilt, since that pattern already
    worked) but logs the richer config/provenance this framework produces:
    git commit, split, K, every active feature weight, fusion method,
    model names, and the full relevance + latency metric set.
    """
    try:
        import mlflow
    except ImportError:
        return False

    config = result["config"]
    summary = result["summary"]
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{config['id']}-{config['split']}"):
        mlflow.log_params(
            {
                "config_id": config["id"],
                "split": config["split"],
                "git_commit": config.get("git_commit", ""),
                "embedding_model": config["embedding_model"],
                "fusion": config["fusion"],
                "normalization": config["normalization"],
                "candidate_k": config["candidate_k"],
                "rerank_depth": config["rerank_depth"],
                "top_k": config["top_k"],
                "reranker_model": config.get("reranker_model"),
                "seed": config.get("seed"),
                **{f"feature_weight__{k}": v for k, v in config["retrieval_features"].items()},
            }
        )
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
    return True

"""Writes the fixed set of experiment manifests to configs/experiments/*.json.

Re-running this script regenerates the same files (no randomness in the
definitions themselves — only the learned-linear model file it points to
was fit with its own recorded seed). Keeping the manifests as data files
(not inline code) is what lets `scripts/run_experiment.py` and the parallel
runner enumerate/select configs without editing source.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LEARNED_MODEL_PATH = str(ROOT / "artifacts" / "models" / "learned_linear_v1.json")

BASE = {
    "embedding_model": ST_MODEL,
    "index_type": "flat",
    "candidate_k": 50,
    "top_k": 5,
    "normalization": "minmax",
    "split": "test",
    "git_commit": "",
}

CONFIGS = {
    # --- baselines (Section 11) ---
    "bm25_only": dict(
        **BASE,
        description="Lexical-only baseline: BM25 score, weighted_sum with weight 1.0.",
        retrieval_features={"bm25": 1.0, "dense": 0.0},
        fusion="weighted_sum",
    ),
    "dense_only": dict(
        **BASE,
        description="Semantic-only baseline: real Sentence-Transformers dense similarity.",
        retrieval_features={"bm25": 0.0, "dense": 1.0},
        fusion="weighted_sum",
    ),
    "hashing_dense_only_ci": dict(
        **{**BASE, "embedding_model": "hashing"},
        description="CI/demo-only baseline using the deterministic hashing embedder (no network, no model download) — NOT used for any headline claim, kept for fast reproducibility checks matching the original repository's default profile.",
        retrieval_features={"bm25": 0.0, "dense": 1.0},
        fusion="weighted_sum",
    ),
    "hybrid_weighted": dict(
        **BASE,
        description="Original repository's fusion strategy, generalized: min-max normalized weighted sum of BM25 + real dense similarity, alpha=0.55 (dense weight), matching the original hardcoded default before any tuning.",
        retrieval_features={"bm25": 0.45, "dense": 0.55},
        fusion="weighted_sum",
    ),
    "hybrid_rrf": dict(
        **BASE,
        description="Same two retrieval signals, fused by reciprocal rank fusion instead of normalized weighted sum (Gate 4 alternative).",
        retrieval_features={"bm25": 1.0, "dense": 1.0},
        fusion="reciprocal_rank_fusion",
    ),
    "hybrid_learned": dict(
        **BASE,
        description="Weighted sum whose weights come from ranking_model.fit_logistic_rank_model trained on the DEV split (machine-learned linear ranking model).",
        retrieval_features={"bm25": 1.0, "dense": 1.0, "title_overlap": 1.0, "freshness": 1.0, "source_authority": 1.0},
        fusion="learned_linear",
        learned_model_path=LEARNED_MODEL_PATH,
    ),
    "cross_encoder_rerank": dict(
        **BASE,
        description="hybrid_weighted retrieval stage + cross-encoder reranking over the top candidates (Gate 1 repair: cross-encoder now genuinely evaluated).",
        retrieval_features={"bm25": 0.45, "dense": 0.55},
        fusion="weighted_sum",
        rerank_depth=20,
        reranker_model=CE_MODEL,
        rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0},
    ),
    # --- full system (candidate "final" configuration, subject to Section 18-20 optimization) ---
    "full": dict(
        **BASE,
        description="All retrieval-stage features (bm25, dense, title_overlap, freshness, source_authority) via weighted_sum, plus cross-encoder reranking. Starting point before validation-only optimization.",
        retrieval_features={
            "bm25": 0.35,
            "dense": 0.45,
            "title_overlap": 0.10,
            "freshness": 0.05,
            "source_authority": 0.05,
        },
        fusion="weighted_sum",
        rerank_depth=20,
        reranker_model=CE_MODEL,
        rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0},
    ),
    # --- ablations (Section 21): each removes exactly one signal from `full` ---
    "full_minus_lexical": dict(
        **BASE,
        description="full system with the BM25 (lexical) weight set to 0.",
        retrieval_features={"bm25": 0.0, "dense": 0.45, "title_overlap": 0.10, "freshness": 0.05, "source_authority": 0.05},
        fusion="weighted_sum", rerank_depth=20, reranker_model=CE_MODEL,
        rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0},
    ),
    "full_minus_dense": dict(
        **BASE,
        description="full system with the dense/semantic weight set to 0.",
        retrieval_features={"bm25": 0.35, "dense": 0.0, "title_overlap": 0.10, "freshness": 0.05, "source_authority": 0.05},
        fusion="weighted_sum", rerank_depth=20, reranker_model=CE_MODEL,
        rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0},
    ),
    "full_minus_cross_encoder": dict(
        **BASE,
        description="full system with the reranking stage disabled (rerank_depth=0).",
        retrieval_features={"bm25": 0.35, "dense": 0.45, "title_overlap": 0.10, "freshness": 0.05, "source_authority": 0.05},
        fusion="weighted_sum", rerank_depth=0,
    ),
    "full_minus_title_overlap": dict(
        **BASE,
        description="full system with the title_overlap weight set to 0.",
        retrieval_features={"bm25": 0.35, "dense": 0.45, "title_overlap": 0.0, "freshness": 0.05, "source_authority": 0.05},
        fusion="weighted_sum", rerank_depth=20, reranker_model=CE_MODEL,
        rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0},
    ),
    "full_minus_freshness": dict(
        **BASE,
        description="full system with the freshness weight set to 0.",
        retrieval_features={"bm25": 0.35, "dense": 0.45, "title_overlap": 0.10, "freshness": 0.0, "source_authority": 0.05},
        fusion="weighted_sum", rerank_depth=20, reranker_model=CE_MODEL,
        rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0},
    ),
    "full_minus_source_authority": dict(
        **BASE,
        description="full system with the source_authority weight set to 0.",
        retrieval_features={"bm25": 0.35, "dense": 0.45, "title_overlap": 0.10, "freshness": 0.05, "source_authority": 0.0},
        fusion="weighted_sum", rerank_depth=20, reranker_model=CE_MODEL,
        rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0},
    ),
}


def main() -> None:
    for config_id, body in CONFIGS.items():
        body = {"id": config_id, **body}
        (OUT / f"{config_id}.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"wrote {len(CONFIGS)} experiment configs -> {OUT}")


if __name__ == "__main__":
    main()

"""Fits the learned-linear ranking model on the DEV split only (never val/test).

For every (query, document) pair in the dev split, computes the full
retrieval-stage feature vector (bm25, dense, title_overlap, freshness,
source_authority) and a binary label (graded relevance > 0), then fits
logistic regression weights via `ranking_model.fit_logistic_rank_model`.
The resulting weights become the `learned_linear` fusion strategy's
coefficients, used only via experiment configs that point at the saved
model file — validation/test evaluation of those configs happens later,
in `evaluate_config`, completely separately from this fitting step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from support_search.embeddings import create_embedder
from support_search.experiment import load_corpus, queries_for_split
from support_search.features import FEATURE_REGISTRY, FeatureContext
from support_search.ranking_model import fit_logistic_rank_model
from support_search.retrieval import BM25Index, DenseIndex

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ["bm25", "dense", "title_overlap", "freshness", "source_authority"]


def main(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    documents = load_corpus()
    embedder = create_embedder(embedding_model)
    bm25 = BM25Index(documents)
    dense = DenseIndex(documents, embedder)

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    for item in queries_for_split("dev"):
        query = item["query"]
        bm25_scores = bm25.scores(query)
        dense_scores = dense.scores(query)
        raw = {"bm25": bm25_scores, "dense": dense_scores}
        relevant_grades = item["relevance"]
        for i, doc in enumerate(documents):
            ctx = FeatureContext(
                query=query,
                document=doc,
                raw_scores={"bm25": float(bm25_scores[i]), "dense": float(dense_scores[i])},
            )
            row = {name: FEATURE_REGISTRY[name].extract(ctx) for name in FEATURE_NAMES}
            rows.append(row)
            labels.append(1 if relevant_grades.get(doc.id, 0) > 0 else 0)

    model = fit_logistic_rank_model(rows, labels, FEATURE_NAMES, seed=42)
    out_dir = ROOT / "artifacts" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "learned_linear_v1.json"
    model.save(out_path)
    print(f"trained on {model.train_rows} (query,doc) rows, positive rate {model.train_positive_rate:.4f}")
    print("coefficients:", json.dumps(model.coefficients(), indent=2))
    print("bias:", model.bias)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

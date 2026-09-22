"""Section 21: feature ablations on the FULL (maximal) architecture, run on
the VALIDATION split (never test) so this diagnostic study never touches
the one held-out read reserved for Section 20's final evaluation.

`full` intentionally includes every implemented signal (bm25, dense,
title_overlap, freshness, source_authority, cross-encoder reranking) even
though the separate validation-only optimization in scripts/optimize.py
already found that most of these do not help on this corpus. Ablating from
`full` (rather than from the already-slimmed-down `final_frozen`) is the
correct way to directly answer "which features actually matter", per
Section 21 — removing a feature from an already-zero-weight config would
not tell us anything.
"""
from __future__ import annotations

import json
from pathlib import Path

from support_search.experiment import ExperimentConfig, evaluate_config, load_corpus, queries_for_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
OUT_DIR = ROOT / "results" / "ablations"

ABLATIONS = [
    "full",
    "full_minus_lexical",
    "full_minus_dense",
    "full_minus_cross_encoder",
    "full_minus_title_overlap",
    "full_minus_freshness",
    "full_minus_source_authority",
]


def main() -> None:
    documents = load_corpus()
    queries = queries_for_split("val")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full_summary = None
    rows = []
    for config_id in ABLATIONS:
        config = ExperimentConfig.load(CONFIG_DIR / f"{config_id}.json")
        config.split = "val"
        result = evaluate_config(config, documents, queries)
        summary = result["summary"]
        (OUT_DIR / f"{config_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if config_id == "full":
            full_summary = summary
        delta_ndcg = summary["ndcg_at_k"] - full_summary["ndcg_at_k"] if full_summary else 0.0
        row = {
            "config_id": config_id,
            "ndcg_at_k": round(summary["ndcg_at_k"], 4),
            "delta_ndcg_vs_full": round(delta_ndcg, 4),
            "mrr": round(summary["mrr"], 4),
            "precision_at_k": round(summary["precision_at_k"], 4),
            "recall_at_k": round(summary["recall_at_k"], 4),
            "p95_latency_ms": round(summary["p95_latency_ms"], 3),
        }
        rows.append(row)
        print(f"{config_id:32s} ndcg@5={row['ndcg_at_k']:.4f}  delta={row['delta_ndcg_vs_full']:+.4f}  p95_ms={row['p95_latency_ms']:7.2f}")

    (OUT_DIR / "ablation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote -> {OUT_DIR / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()

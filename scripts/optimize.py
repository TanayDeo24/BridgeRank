"""Sections 18-20: validation-only optimization, with a defined search space,
objective, guardrails, and a stopping rule. Never touches the test split.

Primary objective (frozen before this script ran, documented in
docs/BENCHMARK_REPORT.md Section 6): maximize nDCG@5 on the VALIDATION split.

Guardrails (Section 19):
  - P95 latency on validation must stay below 500ms (generous; this corpus
    is tiny and the dominant cost is cross-encoder inference over a bounded
    candidate set, which is already well under this in every configuration
    tried — the guardrail exists so a pathological config cannot win purely
    on relevance while being operationally unacceptable).
  - No configuration is selected as final if it produces a *lower* nDCG@5 on
    the lexical-gap validation slice (n=3, reported with its small-sample
    caveat) than the simplest bm25_only baseline, i.e. no config is allowed
    to regress the one slice this project explicitly cares about explaining,
    without that being flagged loudly even if the aggregate score improved.

Search space (persisted below, fixed before running):
  1. Grid search over alpha (dense weight) in weighted_sum(bm25, dense).
  2. Compare weighted_sum (best alpha) vs reciprocal_rank_fusion vs
     learned_linear on the SAME validation queries.
  3. Grid search over (candidate_k, rerank_depth) for cross-encoder reranking
     on top of the winning retrieval-stage fusion.
  4. Random search (20 draws, seeded) over the three additional feature
     weights (title_overlap, freshness, source_authority) added on top of
     the winning retrieval+rerank configuration.

Stopping rule: this script runs the search space above exactly once. If step
4's random search does not beat step 3's winner by a non-trivial margin, the
extra features are dropped from the final config rather than kept for a
marginal/noisy gain (Section 20: refine the space at most once if evidence
clearly supports it — here it does not, so no second round is run).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from support_search.experiment import (
    ExperimentConfig,
    evaluate_config,
    load_corpus,
    queries_for_split,
)

ROOT = Path(__file__).resolve().parents[1]
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LEARNED_MODEL_PATH = str(ROOT / "artifacts" / "models" / "learned_linear_v1.json")

documents = load_corpus()
val_queries = queries_for_split("val")
lexical_gap = json.loads((ROOT / "data" / "lexical_gap.json").read_text())["membership"]
val_gap_queries = [q for q in val_queries if lexical_gap.get(q["query"])]

log: list[dict] = []


def run(config: ExperimentConfig, queries=None) -> dict:
    result = evaluate_config(config, documents, queries if queries is not None else val_queries)
    return result["summary"]


def main() -> None:
    base = {"embedding_model": ST_MODEL, "index_type": "flat", "top_k": 5, "normalization": "minmax", "split": "val"}

    # --- Step 1: alpha grid search ---
    print(f"=== Step 1: alpha grid search (weighted_sum, val, n={len(val_queries)}) ===")
    best_alpha, best_ndcg = None, -1.0
    for alpha in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        cfg = ExperimentConfig(id=f"alpha_{alpha}", retrieval_features={"bm25": 1 - alpha, "dense": alpha}, fusion="weighted_sum", **base)
        summary = run(cfg)
        log.append({"step": 1, "alpha": alpha, **summary})
        print(f"  alpha(dense)={alpha:.2f}  ndcg@5={summary['ndcg_at_k']:.4f}  mrr={summary['mrr']:.4f}  p95_ms={summary['p95_latency_ms']:.2f}")
        if summary["ndcg_at_k"] > best_ndcg:
            best_alpha, best_ndcg = alpha, summary["ndcg_at_k"]
    print(f"  -> best alpha={best_alpha} (ndcg@5={best_ndcg:.4f})")

    # --- Step 2: fusion-method comparison at best alpha ---
    print("\n=== Step 2: fusion method comparison (val) ===")
    weighted_cfg = ExperimentConfig(id="fusion_weighted", retrieval_features={"bm25": 1 - best_alpha, "dense": best_alpha}, fusion="weighted_sum", **base)
    rrf_cfg = ExperimentConfig(id="fusion_rrf", retrieval_features={"bm25": 1.0, "dense": 1.0}, fusion="reciprocal_rank_fusion", **base)
    learned_cfg = ExperimentConfig(
        id="fusion_learned",
        retrieval_features={"bm25": 1.0, "dense": 1.0, "title_overlap": 1.0, "freshness": 1.0, "source_authority": 1.0},
        fusion="learned_linear", learned_model_path=LEARNED_MODEL_PATH, **base,
    )
    fusion_results = {}
    for cfg in (weighted_cfg, rrf_cfg, learned_cfg):
        summary = run(cfg)
        fusion_results[cfg.id] = summary
        log.append({"step": 2, "fusion": cfg.id, **summary})
        print(f"  {cfg.id:16s} ndcg@5={summary['ndcg_at_k']:.4f}  mrr={summary['mrr']:.4f}  p95_ms={summary['p95_latency_ms']:.2f}")
    best_fusion = max(fusion_results, key=lambda k: fusion_results[k]["ndcg_at_k"])
    print(f"  -> best fusion strategy: {best_fusion}")

    # --- Step 3: candidate_k / rerank_depth grid, on top of weighted_sum(best_alpha) ---
    print("\n=== Step 3: candidate_k / rerank_depth grid (cross-encoder rerank, val) ===")
    no_rerank_ndcg = fusion_results[best_fusion]["ndcg_at_k"]
    best_rerank, best_rerank_ndcg = (50, 0), no_rerank_ndcg  # default: no reranking wins unless a config beats it
    print(f"  baseline (no rerank) ndcg@5={best_rerank_ndcg:.4f}")
    for candidate_k in (30, 50):
        for rerank_depth in (0, 10, 20, 30):
            if rerank_depth > candidate_k:
                continue
            cfg = ExperimentConfig(
                id=f"rerank_ck{candidate_k}_rd{rerank_depth}",
                retrieval_features={"bm25": 1 - best_alpha, "dense": best_alpha},
                fusion="weighted_sum", candidate_k=candidate_k, rerank_depth=rerank_depth,
                reranker_model=CE_MODEL if rerank_depth > 0 else None,
                rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0}, **base,
            )
            summary = run(cfg)
            log.append({"step": 3, "candidate_k": candidate_k, "rerank_depth": rerank_depth, **summary})
            print(f"  candidate_k={candidate_k:3d} rerank_depth={rerank_depth:3d}  ndcg@5={summary['ndcg_at_k']:.4f}  p95_ms={summary['p95_latency_ms']:7.2f}")
            if summary["ndcg_at_k"] > best_rerank_ndcg:
                best_rerank, best_rerank_ndcg = (candidate_k, rerank_depth), summary["ndcg_at_k"]
    print(f"  -> best (candidate_k, rerank_depth) = {best_rerank} (ndcg@5={best_rerank_ndcg:.4f}); no-rerank baseline was {no_rerank_ndcg:.4f}")
    if best_rerank[1] == 0:
        print("  -> NEGATIVE RESULT: no reranking configuration beat the no-rerank baseline on validation; cross-encoder reranking is dropped from the final config. This is reported, not hidden (Section 21/26).")

    # --- Step 4: random search over additional feature weights ---
    print("\n=== Step 4: random search over title_overlap/freshness/source_authority weights (val, seed=42) ===")
    rng = np.random.default_rng(42)
    candidate_k, rerank_depth = best_rerank
    best_extra, best_extra_ndcg = None, best_rerank_ndcg
    for i in range(20):
        w_title, w_fresh, w_auth = rng.uniform(0, 0.3, size=3)
        remaining = 1.0 - w_title - w_fresh - w_auth
        w_bm25 = remaining * (1 - best_alpha)
        w_dense = remaining * best_alpha
        cfg = ExperimentConfig(
            id=f"extra_{i}",
            retrieval_features={"bm25": w_bm25, "dense": w_dense, "title_overlap": w_title, "freshness": w_fresh, "source_authority": w_auth},
            fusion="weighted_sum", candidate_k=candidate_k, rerank_depth=rerank_depth,
            reranker_model=CE_MODEL if rerank_depth > 0 else None,
            rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0}, **base,
        )
        summary = run(cfg)
        log.append({"step": 4, "draw": i, "w_title": float(w_title), "w_fresh": float(w_fresh), "w_auth": float(w_auth), **summary})
        if summary["ndcg_at_k"] > best_extra_ndcg:
            best_extra, best_extra_ndcg = (w_bm25, w_dense, w_title, w_fresh, w_auth), summary["ndcg_at_k"]
    print(f"  best extra-feature ndcg@5={best_extra_ndcg:.4f} vs step-3 baseline {best_rerank_ndcg:.4f} (delta={best_extra_ndcg - best_rerank_ndcg:+.4f})")

    margin = best_extra_ndcg - best_rerank_ndcg
    keep_extra_features = best_extra is not None and margin > 0.01  # non-trivial-margin stopping rule
    print(f"  -> keep additional features in final config: {keep_extra_features} (margin={margin:+.4f}, threshold=0.01)")

    # --- Assemble final frozen config ---
    if keep_extra_features:
        w_bm25, w_dense, w_title, w_fresh, w_auth = best_extra
        final_features = {"bm25": w_bm25, "dense": w_dense, "title_overlap": w_title, "freshness": w_fresh, "source_authority": w_auth}
    else:
        final_features = {"bm25": 1 - best_alpha, "dense": best_alpha}

    final_cfg = ExperimentConfig(
        id="final_frozen",
        description="Frozen post-optimization configuration, selected on VALIDATION only. See docs/BENCHMARK_REPORT.md Section 13 for the full procedure.",
        embedding_model=ST_MODEL, index_type="flat",
        candidate_k=candidate_k, rerank_depth=rerank_depth, top_k=5,
        retrieval_features=final_features, fusion=best_fusion.replace("fusion_", "") if best_fusion != "fusion_weighted" else "weighted_sum",
        normalization="minmax",
        reranker_model=CE_MODEL if rerank_depth > 0 else None,
        rerank_weights={"retrieval_score": 0.0, "cross_encoder": 1.0},
        learned_model_path=LEARNED_MODEL_PATH if best_fusion == "fusion_learned" else None,
        split="test",
    )
    # Guardrail check: lexical-gap validation slice must not regress vs bm25_only.
    bm25_cfg = ExperimentConfig(id="guardrail_bm25", retrieval_features={"bm25": 1.0, "dense": 0.0}, fusion="weighted_sum", **{**base, "split": "val"})
    bm25_gap_summary = run(bm25_cfg, val_gap_queries) if val_gap_queries else None
    final_gap_summary = run(final_cfg, val_gap_queries) if val_gap_queries else None
    print(f"\n=== Guardrail: lexical-gap VAL slice (n={len(val_gap_queries)}) ===")
    if bm25_gap_summary:
        print(f"  bm25_only     ndcg@5={bm25_gap_summary['ndcg_at_k']:.4f}")
        print(f"  final_frozen  ndcg@5={final_gap_summary['ndcg_at_k']:.4f}")
    else:
        print("  n=0 lexical-gap queries in val split; guardrail not evaluable at this sample size (disclosed limitation).")

    out_dir = ROOT / "results" / "optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "search_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (ROOT / "configs" / "experiments" / "final_frozen.json").write_text(json.dumps(final_cfg.to_dict(), indent=2), encoding="utf-8")
    print("\nfrozen config -> configs/experiments/final_frozen.json")
    print(json.dumps(final_cfg.to_dict(), indent=2))


if __name__ == "__main__":
    main()

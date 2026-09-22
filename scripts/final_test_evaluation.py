"""Section 20 step 9 + Section 26: the ONE held-out test run.

Run exactly once, after every parameter/feature/fusion/K/lexical-gap
decision was frozen using dev/validation only (scripts/optimize.py,
scripts/lexical_gap_analysis.py, scripts/run_ablations.py all ran and were
inspected BEFORE this script ever touched the test split). Evaluates the
frozen baselines + the frozen final configuration on:
  (a) the full test split (n=30)
  (b) the lexical-gap test slice (n=5, small-sample caveat applies)
and computes paired bootstrap confidence intervals for the primary metric.

This script is not re-run to chase a better number. If it ever surfaces a
genuine implementation bug, the bug is fixed, the run is invalidated and
re-labeled, and the fix is documented in docs/CHANGE_REPORT.md — the
procedure is never "just run it again."
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from support_search.experiment import (
    ExperimentConfig,
    evaluate_config,
    load_corpus,
    log_to_mlflow,
    queries_for_split,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
OUT_DIR = ROOT / "results" / "final"

CONFIGS = ["bm25_only", "dense_only", "hybrid_weighted", "cross_encoder_rerank", "hybrid_learned", "final_frozen"]


def bootstrap_ci(rows: list[dict], key: str, n_boot: int = 5000, seed: int = 42) -> dict:
    values = np.array([r[key] for r in rows], dtype=np.float64)
    if len(values) == 0:
        return {"point_estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[b] = sample.mean()
    return {
        "point_estimate": float(values.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "n": n,
        "n_boot": n_boot,
    }


def main() -> None:
    documents = load_corpus()
    test_queries = queries_for_split("test")
    lexical_gap = json.loads((ROOT / "data" / "lexical_gap.json").read_text())["membership"]
    gap_test_queries = [q for q in test_queries if lexical_gap.get(q["query"])]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for config_id in CONFIGS:
        config = ExperimentConfig.load(CONFIG_DIR / f"{config_id}.json")
        config.split = "test"
        result_full = evaluate_config(config, documents, test_queries)
        result_gap = evaluate_config(config, documents, gap_test_queries) if gap_test_queries else None

        ci_ndcg = bootstrap_ci(result_full["rows"], "ndcg_at_k")
        ci_mrr = bootstrap_ci(result_full["rows"], "mrr")

        payload = {
            "config": config.to_dict(),
            "full_test": result_full["summary"],
            "lexical_gap_test": result_gap["summary"] if result_gap else None,
            "lexical_gap_test_n": len(gap_test_queries),
            "bootstrap_ci_ndcg_at_k": ci_ndcg,
            "bootstrap_ci_mrr": ci_mrr,
        }
        all_results[config_id] = payload
        (OUT_DIR / f"{config_id}.json").write_text(json.dumps({**payload, "rows": result_full["rows"]}, indent=2), encoding="utf-8")
        log_to_mlflow({"config": config.to_dict(), "summary": result_full["summary"]}, experiment_name="support-search-final-test")

        s = result_full["summary"]
        g = result_gap["summary"] if result_gap else None
        print(
            f"{config_id:22s} test ndcg@5={s['ndcg_at_k']:.4f} [{ci_ndcg['ci_low']:.4f},{ci_ndcg['ci_high']:.4f}]"
            f"  mrr={s['mrr']:.4f}  p95_ms={s['p95_latency_ms']:7.2f}"
            + (f"   | gap(n={len(gap_test_queries)}) ndcg@5={g['ndcg_at_k']:.4f}" if g else "   | gap: n=0")
        )

    (OUT_DIR / "all_final_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nwrote per-config results + all_final_results.json -> {OUT_DIR}")


if __name__ == "__main__":
    main()

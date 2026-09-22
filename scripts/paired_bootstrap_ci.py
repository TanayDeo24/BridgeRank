"""Closure-pass addition: paired bootstrap CI for the final-vs-BM25 nDCG@5
delta on the held-out test split.

This is a PURELY OBSERVATIONAL script. It reads two already-frozen,
already-committed result files (`results/final/bm25_only.json` and
`results/final/final_frozen.json`), both produced by the one held-out test
run described in `docs/BENCHMARK_REPORT.md`/`docs/FINAL_CLOSURE_REPORT.md`.
It does not call any ranking code, does not touch the frozen config, does
not change the test population, does not change K, and does not make any
tuning decision. It only resamples the already-stored per-query nDCG@5
values that both files' `rows` arrays already contain, paired by query
text (verified identical order and identical 30-query set in both files
before running).

Method: paired bootstrap. Resample query *pairs* (bm25_ndcg[i], final_ndcg[i])
with replacement, 5000 times, seed=42; for each resample compute the mean of
(final - bm25) across the resampled indices; report the 2.5/97.5 percentiles
of that distribution as the 95% CI for the true mean paired delta.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 5000
SEED = 42


def main() -> None:
    bm25 = json.loads((ROOT / "results" / "final" / "bm25_only.json").read_text())
    frozen = json.loads((ROOT / "results" / "final" / "final_frozen.json").read_text())

    bm25_rows = {r["query"]: r["ndcg_at_k"] for r in bm25["rows"]}
    frozen_rows = {r["query"]: r["ndcg_at_k"] for r in frozen["rows"]}

    assert set(bm25_rows) == set(frozen_rows), "query sets differ -- cannot pair"
    queries = sorted(bm25_rows)  # deterministic order, independent of file order
    assert len(queries) == 30, f"expected 30 held-out test queries, found {len(queries)}"

    bm25_vals = np.array([bm25_rows[q] for q in queries], dtype=np.float64)
    frozen_vals = np.array([frozen_rows[q] for q in queries], dtype=np.float64)
    deltas = frozen_vals - bm25_vals

    observed_delta = float(deltas.mean())

    rng = np.random.default_rng(SEED)
    n = len(deltas)
    boot_means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)  # resample query PAIRS (same idx into both arrays)
        boot_means[b] = deltas[idx].mean()

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    includes_zero = ci_low <= 0.0 <= ci_high

    result = {
        "method": "paired bootstrap, resampling query pairs with replacement",
        "source_files": [
            "results/final/bm25_only.json",
            "results/final/final_frozen.json",
        ],
        "config_a": "bm25_only",
        "config_b": "final_frozen",
        "metric": "ndcg_at_k (nDCG@5)",
        "split": "test",
        "query_count": n,
        "seed": SEED,
        "n_boot": N_BOOT,
        "observed_mean_delta": observed_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "includes_zero": includes_zero,
    }
    out_path = ROOT / "results" / "final" / "paired_bootstrap_ndcg_delta.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"n={n} queries, seed={SEED}, n_boot={N_BOOT}")
    print(f"observed mean delta (final - bm25) nDCG@5 = {observed_delta:.6f}")
    print(f"95% paired bootstrap CI = [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"CI includes zero: {includes_zero}")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()

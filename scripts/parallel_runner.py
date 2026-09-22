"""Actual parallel ranking-experiment tooling (Section 22) + benchmark protocol (Section 23).

Given a manifest (a list of experiment config paths + the split to run them
against), this:

1. enumerates the configs,
2. runs them with a configurable worker-pool size using
   ``concurrent.futures.ProcessPoolExecutor`` (real OS-process parallelism,
   not just asyncio concurrency over one interpreter — each worker loads its
   own copy of the embedding/reranker models, which is exactly the kind of
   per-worker cost a genuine parallel benchmark needs to include),
3. isolates each config's output file so failures in one config never
   clobber another's results,
4. captures per-config success/failure and wall time,
5. writes one aggregated comparison table (JSON + CSV) per worker-count run,
   plus a final speedup/throughput summary across worker counts.

Usage:

    python scripts/parallel_runner.py --manifest configs/manifests/all_test.json \
        --workers 1 2 4 --out results/parallel_experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from support_search.experiment import (
    ExperimentConfig,
    evaluate_config,
    load_corpus,
    queries_for_split,
)

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - git metadata is best-effort, never fatal
        return "unknown"


@dataclass
class RunOutcome:
    config_id: str
    split: str
    success: bool
    wall_time_s: float
    error: str | None
    summary: dict | None


def _run_one(config_path: str, split: str, out_dir: str) -> RunOutcome:
    """Top-level (picklable) worker function executed in a subprocess.

    Everything, including config loading itself, is inside the try block:
    a missing/malformed config file must surface as one failed row in the
    comparison table, not crash the whole manifest (Section 29: "failed-
    worker handling" is a tested property, not an assumption)."""
    started = time.perf_counter()
    config_id_for_error = Path(config_path).stem
    try:
        config = ExperimentConfig.load(Path(config_path))
        config.split = split
        documents = load_corpus()
        queries = queries_for_split(split)
        result = evaluate_config(config, documents, queries)
        wall = time.perf_counter() - started
        out_path = Path(out_dir) / f"{config.id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return RunOutcome(config.id, split, True, wall, None, result["summary"])
    except Exception as exc:  # noqa: BLE001 - a failed config run is a measured outcome
        wall = time.perf_counter() - started
        return RunOutcome(config_id_for_error, split, False, wall, f"{type(exc).__name__}: {exc}", None)


def run_manifest(config_paths: list[Path], split: str, workers: int, out_dir: Path) -> tuple[list[RunOutcome], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outcomes: list[RunOutcome] = []
    if workers <= 1:
        for path in config_paths:
            outcomes.append(_run_one(str(path), split, str(out_dir)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, str(path), split, str(out_dir)): path for path in config_paths}
            for future in as_completed(futures):
                outcomes.append(future.result())
    wall_clock = time.perf_counter() - started
    return outcomes, wall_clock


def write_comparison(outcomes: list[RunOutcome], workers: int, wall_clock: float, out_dir: Path) -> dict:
    rows = []
    for o in sorted(outcomes, key=lambda x: x.config_id):
        row = {
            "config_id": o.config_id,
            "split": o.split,
            "success": o.success,
            "wall_time_s": round(o.wall_time_s, 4),
            "error": o.error,
        }
        if o.summary:
            row.update(
                {
                    "ndcg_at_k": round(o.summary["ndcg_at_k"], 4),
                    "mrr": round(o.summary["mrr"], 4),
                    "precision_at_k": round(o.summary["precision_at_k"], 4),
                    "recall_at_k": round(o.summary["recall_at_k"], 4),
                    "failed_query_rate": round(o.summary["failed_query_rate"], 4),
                    "p95_latency_ms": round(o.summary["p95_latency_ms"], 4),
                }
            )
        rows.append(row)

    report = {
        "workers": workers,
        "config_count": len(outcomes),
        "wall_clock_s": round(wall_clock, 4),
        "configs_per_hour": round(len(outcomes) / wall_clock * 3600, 2) if wall_clock > 0 else None,
        "failures": [o.config_id for o in outcomes if not o.success],
        "rows": rows,
    }
    (out_dir / f"comparison_workers_{workers}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out_dir / f"comparison_workers_{workers}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path, help="JSON list of experiment config paths")
    parser.add_argument("--split", default="test", choices=["dev", "val", "test"])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "parallel_experiments")
    args = parser.parse_args()

    config_paths = [Path(p) for p in json.loads(args.manifest.read_text(encoding="utf-8"))]
    commit = git_commit()
    print(f"manifest: {len(config_paths)} configs, split={args.split}, commit={commit}")
    print(f"machine: cpu_count={os.cpu_count()}")

    summary_rows = []
    for workers in args.workers:
        outcomes, wall_clock = run_manifest(config_paths, args.split, workers, args.out / f"workers_{workers}")
        report = write_comparison(outcomes, workers, wall_clock, args.out)
        print(f"workers={workers:>2}  wall_clock={wall_clock:8.3f}s  configs/hour={report['configs_per_hour']:>10}  failures={report['failures']}")
        summary_rows.append(
            {"workers": workers, "wall_clock_s": round(wall_clock, 4), "configs_per_hour": report["configs_per_hour"], "failures": len(report["failures"])}
        )

    sequential = next((r for r in summary_rows if r["workers"] == 1), None)
    if sequential:
        for row in summary_rows:
            row["speedup_vs_sequential"] = (
                round(sequential["wall_clock_s"] / row["wall_clock_s"], 3) if row["wall_clock_s"] > 0 else None
            )

    final = {
        "manifest": str(args.manifest),
        "split": args.split,
        "git_commit": commit,
        "cpu_count": os.cpu_count(),
        "config_count": len(config_paths),
        "runs": summary_rows,
    }
    (args.out / "speedup_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

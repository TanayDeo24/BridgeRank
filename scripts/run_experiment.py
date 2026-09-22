"""CLI entry point for running one experiment config against one data split.

    python scripts/run_experiment.py --config configs/experiments/hybrid_weighted.json --split val
    python scripts/run_experiment.py --config configs/experiments/full.json --split test --out results/final/full_test.json

The `--split` flag always overrides the config file's own `split` field, so
the same frozen config can be pointed at dev/val/test at different phases
without editing the file (Section 20: the config's parameters are frozen;
only which split it is *measured against* changes across phases).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from support_search.experiment import ExperimentConfig, evaluate_config, load_corpus, log_to_mlflow, queries_for_split


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=["dev", "val", "test"], default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    if args.split:
        config.split = args.split
    config.git_commit = git_commit()

    documents = load_corpus()
    queries = queries_for_split(config.split)
    result = evaluate_config(config, documents, queries)

    out_path = args.out or Path("results") / config.split / f"{config.id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"config_id": config.id, "split": config.split, "summary": result["summary"]}, indent=2))
    print(f"wrote -> {out_path}")
    print(f"mlflow logged: {log_to_mlflow(result)}")


if __name__ == "__main__":
    main()

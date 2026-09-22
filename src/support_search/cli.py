from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import build_artifact
from .config import Settings
from .embeddings import create_embedder
from .evaluation import evaluate
from .failure_analysis import create_summary
from .io import load_documents, load_queries
from .service import SearchService


def run_evaluation(args) -> None:
    settings = Settings(data_path=args.documents, artifact_dir=Path("__missing_artifact__"))
    service = SearchService(settings)
    queries = load_queries(args.queries)
    all_metrics = {}
    for mode in args.modes:
        metrics = evaluate(service.engine, queries, args.top_k, mode)
        all_metrics[mode] = metrics
        try:
            import mlflow

            mlflow.set_experiment("support-search-retrieval")
            with mlflow.start_run(run_name=f"{mode}-{service.embedder.name}"):
                mlflow.log_params(
                    {"mode": mode, "top_k": args.top_k, "embedding_model": service.embedder.name}
                )
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                )
        except ImportError:
            pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))


def run_build_index(args) -> None:
    documents = load_documents(args.documents)
    manifest = build_artifact(documents, create_embedder(), Path(args.documents), args.output)
    print(json.dumps(manifest, indent=2))


def run_quality_gate(args) -> None:
    metrics = json.loads(args.input.read_text(encoding="utf-8"))[args.mode]
    failures = []
    if metrics["ndcg_at_k"] < args.min_ndcg:
        failures.append(f"nDCG {metrics['ndcg_at_k']:.3f} < {args.min_ndcg:.3f}")
    if metrics["mrr"] < args.min_mrr:
        failures.append(f"MRR {metrics['mrr']:.3f} < {args.min_mrr:.3f}")
    if metrics["p95_latency_ms"] > args.max_p95_ms:
        failures.append(f"P95 {metrics['p95_latency_ms']:.1f} ms > {args.max_p95_ms:.1f} ms")
    if metrics["failed_query_rate"] > args.max_failed_rate:
        failures.append(
            f"failed rate {metrics['failed_query_rate']:.1%} > {args.max_failed_rate:.1%}"
        )
    if failures:
        print("Quality gate failed: " + "; ".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(f"Quality gate passed for {args.mode}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="support-search")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="benchmark retrieval strategies")
    ev.add_argument("--documents", default=root / "data/sample_documents.jsonl")
    ev.add_argument("--queries", default=root / "data/evaluation_queries.jsonl")
    ev.add_argument("--output", type=Path, default=root / "artifacts/evaluation.json")
    ev.add_argument("--top-k", type=int, default=3)
    ev.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    ev.set_defaults(func=run_evaluation)
    build = sub.add_parser("build-index", help="build a checksummed immutable index artifact")
    build.add_argument("--documents", default=root / "data/sample_documents.jsonl")
    build.add_argument("--output", type=Path, default=root / "artifacts/index/current")
    build.set_defaults(func=run_build_index)
    gate = sub.add_parser("quality-gate", help="enforce release quality and latency thresholds")
    gate.add_argument("--input", type=Path, default=root / "artifacts/evaluation.json")
    gate.add_argument("--mode", default="hybrid")
    gate.add_argument("--min-ndcg", type=float, default=0.65)
    gate.add_argument("--min-mrr", type=float, default=0.65)
    gate.add_argument("--max-p95-ms", type=float, default=150.0)
    gate.add_argument("--max-failed-rate", type=float, default=0.35)
    gate.set_defaults(func=run_quality_gate)
    report = sub.add_parser("failure-report", help="create a Claude-assisted evaluation summary")
    report.add_argument("--input", type=Path, default=root / "artifacts/evaluation.json")
    report.add_argument("--output", type=Path, default=root / "artifacts/failure_analysis.md")
    report.set_defaults(func=lambda args: create_summary(args.input, args.output))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# Project Evidence

Factual engineering evidence, each statement pointing to its source artifact. Not a resume-bullet
file — see `docs/BENCHMARK_REPORT.md` and `docs/CHANGE_REPORT.md` for full context.

| Fact | Value | Source |
|---|---|---|
| Ranking feature families supported | 6 (`bm25`, `dense`, `title_overlap`, `freshness`, `source_authority`, `cross_encoder`) | `src/support_search/features.py:FEATURE_REGISTRY` |
| Ranking stages | 2 (first-stage retrieval: BM25+dense; second-stage reranking: cross-encoder) | `docs/FINAL_ARCHITECTURE.md` |
| Fusion strategies implemented and compared | 3 (`weighted_sum`, `reciprocal_rank_fusion`, `learned_linear`) | `src/support_search/fusion.py`; comparison in `results/optimization/search_log.json` step 2 |
| Final selected ranking configuration | `bm25` weight 0.30 / `dense` weight 0.70, `weighted_sum`, no reranking, no extra features | `configs/experiments/final_frozen.json` |
| Primary relevance metric | nDCG@5 | `docs/BENCHMARK_REPORT.md` Section 6 |
| Primary K | 5 | `docs/BENCHMARK_REPORT.md` Section 5 |
| BM25 baseline (test, n=30) | nDCG@5 = 0.8673 [0.7462, 0.9667] | `results/final/bm25_only.json` |
| Final held-out performance (test, n=30) | nDCG@5 = 0.9871 [0.9619, 1.0000], MRR = 0.9833, 0% failed | `results/final/final_frozen.json` |
| Lexical-gap slice size | 31/150 queries (20.7%): 23 dev / 3 val / 5 test | `data/lexical_gap.json` |
| Lexical-gap baseline (BM25, test n=5) | nDCG@5 = 0.4036 | `results/final/bm25_only.json` (`lexical_gap_test`) |
| Lexical-gap final (test, n=5) | nDCG@5 = 0.9262 | `results/final/final_frozen.json` (`lexical_gap_test`) |
| Strongest feature-ablation finding | Removing cross-encoder reranking is the only ablation that *improves* the system: +0.0183 nDCG@5, ~6x lower P95 latency | `results/ablations/ablation_summary.json` |
| P95 ranking latency, final config | 6.57 ms (in-process, single request, warm) | `results/latency/final_frozen.json` |
| P95 ranking latency, with cross-encoder reranking | 36.99 ms (~5.6x the non-reranking figure) | `results/latency/cross_encoder_rerank.json` |
| Parallel experiment speedup | 1.43x at 4 workers (peak); 0.93x at 8 workers on an 8-core machine (disclosed regression, not hidden) | `results/parallel_experiments/speedup_summary.json` |
| Configurations per hour | 2413 (1 worker) → 3451 (4 workers, peak) | `results/parallel_experiments/speedup_summary.json` |
| Query count | 150 total (90 dev / 30 val / 30 test) | `data/evaluation_queries.jsonl`, `data/splits.json` |
| Document count | 150 (10 categories x 15) | `data/sample_documents.jsonl` |
| Test status | 44/44 tests pass; `ruff check src tests loadtest dashboard scripts` clean | run `pytest -q` and `ruff check ...` per `docs/REPRODUCIBILITY.md` |
| Negative results reported | 5, itemized | `docs/BENCHMARK_REPORT.md` Section 17 |

# Change Report

## ORIGINAL CAPABILITY (present before this project, unchanged in behavior)

| Capability | Files |
|---|---|
| Hand-rolled Okapi BM25 | `src/support_search/retrieval.py:BM25Index` |
| Dense retrieval (FAISS optional, hashing/ST embedders) | `retrieval.py:DenseIndex`, `embeddings.py` |
| Weighted hybrid fusion (min-max + weighted sum) | `retrieval.py:HybridSearchEngine` |
| Cross-encoder wrapper | `retrieval.py:CrossEncoderReranker` |
| Ranking metrics (P@K, R@K, MRR, nDCG, percentile) | `evaluation.py` |
| FastAPI service, health probes, Prometheus metrics | `api.py`, `service.py` |
| Immutable checksummed index artifacts | `artifacts.py` |
| PostgreSQL/SQLite event + feedback store | `storage.py` |
| Streamlit dashboard | `dashboard/app.py` |
| Docker/Kubernetes/CI/Locust | `Dockerfile`, `ops/kubernetes/`, `.github/workflows/`, `loadtest/` |
| Original 10-document/10-query corpus | `data/sample_documents_v1.jsonl`, `data/evaluation_queries_v1.jsonl` (preserved, never modified) |
| MLflow best-effort logging in the CLI | `cli.py:run_evaluation` |

## REPAIRED CAPABILITY (existed but was invalid/incomplete; fixed, not rebuilt)

| What changed | Why | Files | Measurable effect |
|---|---|---|---|
| Cross-encoder now genuinely evaluated (Gate 1) | Previously wired only to the live API, never to `evaluate()`; zero evidence it helped or hurt | `experiment.py:score_query` (new evaluation path); `retrieval.py:CrossEncoderReranker` unchanged | Found it *reduces* validation nDCG@5 (0.9847→0.9655) — a real, previously-unknown finding |
| Score fusion generalized + alternatives compared (Gate 4) | Original `alpha=0.55` was a hardcoded, never-tuned constant; no alternative fusion method existed | `fusion.py` (new), `_normalize` logic ported from `retrieval.py` | Validation-optimal weighted-sum alpha found to be 0.70, not 0.55; RRF and learned-linear compared and rejected with evidence |
| Persistent, seeded, leakage-free splits (Gate 3) | No split existed; every number used the same 10 queries for everything | `scripts/build_splits.py`, `data/splits.json` | Enabled every validation-only optimization/ablation step that follows |
| `sentence-transformers/all-MiniLM-L6-v2` get_embedding_dimension deprecation | Minor forward-compat bug surfaced while testing | `embeddings.py:SentenceTransformerEmbedder.__init__` | No behavior change, removes a `FutureWarning` |
| Removed unused `rank-bm25` dependency | Declared in `pyproject.toml` but never imported anywhere in `src/` (BM25 is hand-rolled) | `pyproject.toml` | Hygiene only |
| `_run_one` in the parallel runner now handles a missing/malformed config file | Found via test-writing (Section 29): config loading happened outside the try/except, so a bad config path crashed the entire manifest instead of being reported as one failed row | `scripts/parallel_runner.py` | `tests/test_parallel_runner.py::test_run_one_captures_failure_from_a_missing_config_file` now passes; previously would have raised `BrokenProcessPool` |
| `build_engine_bundle`'s cache key now includes a document-set signature | Found via test-writing: the cache was keyed only on `(embedding_model, index_type)`, so two different document sets sharing those would silently share a stale index | `experiment.py:build_engine_bundle` | Prevents a latent correctness bug from ever surfacing in multi-corpus test scenarios; no effect on any single-corpus script in this repo, all of which use one fixed corpus per process |
| `optimize.py`'s candidate_k/rerank_depth search initially defaulted to a reranking config even when no reranking configuration won | Found while inspecting the very first optimization run's printed output (the "best" tuple stayed `None` because nothing beat the no-rerank baseline, but the assembly code silently fell back to `(50, 20)` — i.e. reranking *on* — instead of the correct default of *no reranking*) | `scripts/optimize.py` | Re-ran after the fix; `final_frozen.json` correctly has `reranker_model: null`. The original (buggy) run was invalidated and never used for any reported number (Section 20's rule: fix the bug, invalidate the run, re-establish a clean procedure) |

## NEWLY IMPLEMENTED CAPABILITY

| Capability | Files | Section |
|---|---|---|
| Expanded, disclosed, self-authored 150-doc/150-query corpus | `data/sample_documents.jsonl`, `data/evaluation_queries.jsonl`, `scripts/generate_corpus.py`, `docs/DATA_PROVENANCE.md` | Gate 2 |
| Deterministic seeded query-level stratified split | `scripts/build_splits.py`, `data/splits.json` | Gate 3 |
| Diversified, pluggable ranking-feature framework | `src/support_search/features.py` | 12, 14 |
| Two new ranking-feature families (freshness, source_authority) | `features.py` | 13 |
| Three fusion strategies (weighted_sum generalized, RRF, learned_linear) | `src/support_search/fusion.py` | Gate 4, 14 |
| Machine-learned linear ranking model (logistic regression, dependency-free) | `src/support_search/ranking_model.py`, `scripts/train_learned_model.py` | 8, 18 |
| Configurable experiment framework + CLI | `src/support_search/experiment.py`, `scripts/run_experiment.py`, `configs/experiments/*.json` | 14-16 |
| Lexical-gap definition analysis + frozen membership | `scripts/lexical_gap_analysis.py`, `data/lexical_gap.json`, `docs/LEXICAL_GAP_METHODOLOGY.md` | 17 |
| Validation-only optimization procedure | `scripts/optimize.py`, `results/optimization/` | 18-20 |
| Feature ablation study | `scripts/run_ablations.py`, `results/ablations/` | 21 |
| Real parallel experiment runner (ProcessPoolExecutor) | `scripts/parallel_runner.py` | 22 |
| Parallel benchmark protocol across worker counts | `results/parallel_experiments/` | 23 |
| Latency benchmark protocol (warmup, fixed set, P50/P95/P99, hardware/software recorded) | `scripts/latency_benchmark.py`, `results/latency/` | 24 |
| Single frozen held-out test evaluation + bootstrap CIs | `scripts/final_test_evaluation.py`, `results/final/` | 20, 26 |
| Sourced 10-category failure analysis | `docs/FAILURE_ANALYSIS.md` | 27 |
| MLflow logging wired into the new experiment framework | `experiment.py:log_to_mlflow` | 28 |
| 34 new tests (features, fusion, config, splits, lexical-gap, ranking model, parallel runner) | `tests/test_features.py`, `tests/test_fusion.py`, `tests/test_experiment.py`, `tests/test_data_integrity.py`, `tests/test_ranking_model.py`, `tests/test_parallel_runner.py` | 29 |
| Full documentation set | `docs/PROJECT_STATE_INITIAL.md`, `docs/SEARCH_RELEVANCE_AUDIT.md`, `docs/DATA_PROVENANCE.md`, `docs/LEXICAL_GAP_METHODOLOGY.md`, `docs/FEATURE_INVENTORY.md`, `docs/FINAL_ARCHITECTURE.md`, `docs/BENCHMARK_REPORT.md`, `docs/FAILURE_ANALYSIS.md`, `docs/REPRODUCIBILITY.md`, `docs/CHANGE_REPORT.md`, `docs/PROJECT_EVIDENCE.md` | 30-36 |

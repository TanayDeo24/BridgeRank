# Project State — Initial Snapshot

Recorded before any modification, per the mandatory pre-work safety check.

## Repository provenance

- **Source:** https://github.com/tanaydeo/hybrid-support-search
- **Acquisition method:** the working directory was `hybrid-support-search-main`, the exact
  naming pattern GitHub produces for a "Download ZIP" of the `main` branch — **not** a `git clone`.
  `git status` at task start returned `fatal: not a git repository`. There was no `.git` directory
  and therefore no prior commit hash to record.
- **Action taken:** ran `git init` and made one commit (`93ed1b8`) containing the repository exactly
  as received, before any other change. This commit is the baseline for every diff going forward.
  No history was rewritten because none existed; nothing was overwritten or discarded.
- **Branch:** `main` (newly created by `git init`; not pushed anywhere).
- **Baseline commit:** `93ed1b8` — "Initial import: snapshot of tanaydeo/hybrid-support-search
  (main, downloaded as zip, no prior git history)"
- **Uncommitted user changes at start:** none (nothing existed to be uncommitted; the entire tree
  was untracked until the baseline commit above).
- No push to any remote has been performed or will be performed without explicit request.

## Runtime environment

- **Python (interpreter used for this session):** 3.12.7
- **Declared requirement (`pyproject.toml`):** `requires-python = ">=3.10"`
- **Installed in the ambient environment at audit time (partial, ambient — not yet a project venv):**
  `numpy 1.26.4`, `pydantic 2.12.5`, `pydantic-settings 2.6.1`, `torch 2.7.0`. FastAPI, rank-bm25,
  sentence-transformers, faiss-cpu, mlflow and the `dev` test toolchain were **not** present in the
  ambient environment and must be installed via `pip install -e ".[dev]"` / `".[dev,ml,dashboard]"`
  before tests or benchmarks can run. No project-local virtualenv existed yet.
- **OS:** Darwin 24.6.0 (macOS), shell `zsh`.

## Dependency manifest (`pyproject.toml`)

| Group | Packages |
|---|---|
| core | fastapi, uvicorn, numpy, pydantic, pydantic-settings, prometheus-client, rank-bm25, sqlalchemy, psycopg |
| `ml` (optional) | faiss-cpu, sentence-transformers, torch, transformers, mlflow, anthropic, pandas |
| `dev` (optional) | pytest, pytest-cov, httpx, ruff, locust, pre-commit |
| `dashboard` (optional) | streamlit, pandas, plotly |

**Discrepancy noted (carried into the audit):** `rank-bm25` is declared as a core dependency but is
never imported anywhere in `src/`. `src/support_search/retrieval.py` implements its own hand-rolled
`BM25Index` (Okapi BM25, k1=1.5, b=0.75) instead of using the `rank_bm25` library. This is not
necessarily wrong, but it is a real discrepancy between the manifest and the code.

## Models referenced in source/config (not necessarily active by default)

- Lexical: custom in-repo Okapi BM25 (`retrieval.py:BM25Index`) — always active, no external model.
- Dense/default: `HashingEmbedder` — deterministic blake2b hashed unigram+bigram bag, 384-dim,
  **no pretrained model, no network dependency**. This is the default (`SEARCH_EMBEDDING_MODEL=hashing`).
- Dense/optional: `SentenceTransformerEmbedder`, defaulting to
  `sentence-transformers/all-MiniLM-L6-v2` when `SEARCH_EMBEDDING_MODEL` is set to a HF model id.
  Requires the `ml` extra.
- Reranker/optional: `CrossEncoderReranker` wrapping `cross-encoder/ms-marco-MiniLM-L-6-v2` (or any
  configured `SEARCH_RERANKER_MODEL`) via `sentence_transformers.CrossEncoder`. Only instantiated if
  `SEARCH_RERANKER_MODEL` is set; `None` otherwise.
- Optional: `anthropic` SDK for `failure-report` (`failure_analysis.py`), used only if
  `ANTHROPIC_API_KEY` is set; otherwise the CLI writes a deterministic offline prompt file instead of
  calling any API.

## Existing datasets

- `data/sample_documents.jsonl` — **10 documents**, one line per doc: `id, title, text, category,
  product, metadata.difficulty`. Categories: network(2), account(2), hardware(2), storage(1),
  software(1), collaboration(1), messaging(1).
- `data/evaluation_queries.jsonl` — **10 queries**, one line per query: `query, relevance
  {doc_id: grade}, category`. Graded 0–3 relevance scale (grade 3 assigned to 10/10 queries' primary
  target; 1 secondary relevant doc for 5/10 queries). Total judged pairs: 15. Relevant docs per query
  range 1–2 (mean 1.5). Categories mirror the document categories exactly (1:1 query:doc-category
  ratio), consistent with a hand-authored toy set rather than a sampled query log.
- No train/validation/test split exists anywhere in the repository. The same 10 queries are used for
  every reported number in the README.

## Repository directory structure

```
.
├── src/support_search/
│   ├── api.py               FastAPI app: /v1/search, /v1/feedback, /health/live, /health/ready, /metrics
│   ├── artifacts.py         checksummed, atomically-promoted index bundle (build_artifact/load_artifact)
│   ├── cli.py                `support-search {build-index,evaluate,quality-gate,failure-report}`
│   ├── config.py             pydantic-settings Settings (env prefix SEARCH_)
│   ├── embeddings.py         HashingEmbedder (default) / SentenceTransformerEmbedder
│   ├── evaluation.py         precision/recall/MRR/nDCG + latency percentiles + category slices
│   ├── failure_analysis.py   optional Claude-assisted summary of evaluation.json
│   ├── io.py                 JSONL document/query loaders
│   ├── models.py             Document / SearchResult dataclasses
│   ├── retrieval.py          BM25Index, DenseIndex (FAISS optional), HybridSearchEngine, CrossEncoderReranker
│   ├── service.py            SearchService orchestration + audit-event recording
│   └── storage.py            SQLite/PostgreSQL event + feedback store
├── data/                     sample_documents.jsonl, evaluation_queries.jsonl (10 + 10)
├── dashboard/app.py          Streamlit monitoring dashboard
├── loadtest/locustfile.py    Locust load test
├── ops/kubernetes/           Deployment, Service, HPA, PDB, Kustomization
├── tests/                    test_api.py, test_artifacts.py, test_evaluation_pipeline.py, test_metrics.py, test_retrieval.py
├── docs/architecture.md      existing design doc
├── .github/workflows/        CI (lint/test/build referenced in README; not yet inspected line-by-line)
├── Dockerfile, docker-compose.yml, Makefile, pyproject.toml
├── MODEL_CARD.md, SECURITY.md, CONTRIBUTING.md
```

## Existing test commands (from README / pyproject)

```bash
pytest -q
ruff check src tests loadtest dashboard
pytest --cov=support_search --cov-report=term-missing   # fail_under = 70 in pyproject
```

## Existing benchmark commands (from README / cli.py)

```bash
support-search build-index
support-search evaluate            # writes artifacts/evaluation.json, modes: bm25 dense hybrid, K default 3 from CLI (evaluation.py default K=5)
support-search quality-gate        # thresholds: min_ndcg=0.65 min_mrr=0.65 max_p95_ms=150 max_failed_rate=0.35, mode=hybrid
support-search failure-report
```

Note: `cli.py`'s `evaluate` subcommand passes `--top-k` default `3`, while `evaluation.py::evaluate()`'s
own function signature defaults to `k=5` — the CLI default is what actually governs the README's
reported `K=3` benchmark; the function default of 5 is dead unless `evaluate()` is called directly.

## Current README claims (to be verified against code, not assumed true)

- Claims a working BM25 + dense + hybrid pipeline with configurable fusion weight — **confirmed in code**
  (`retrieval.py`).
- Claims cross-encoder reranking as part of the pipeline — **code exists but, per the Gate 1 audit below,
  is not exercised by the evaluation/benchmark path that produced the reported numbers.**
- Reports: BM25 `P@3=0.300 R@3=0.650 MRR=0.733 nDCG@3=0.725`, Dense `0.300/0.550/0.550/0.513`,
  Hybrid `0.333/0.650/0.700/0.690`, all on the 10-query set with the default `HashingEmbedder` (the
  README explicitly and correctly labels this "reproducible lightweight benchmark," not a claim about
  production performance) — **plausible and reproducible given the code**, not yet independently
  re-run in this session.
- Claims MLflow, Prometheus, PostgreSQL, Docker/Kubernetes, Locust, pre-commit/CI integration — present
  as code/config; operational validity not yet exercised in this session.
- README's own "Limitations and next steps" section already states the corpus is "intentionally small"
  and recommends "at least 500 representative queries with independently reviewed graded judgments" —
  i.e., the repository's own author already flags the exact Gate 2 deficiency identified independently
  in this audit.

This document will not be edited retroactively; corrections belong in `docs/SEARCH_RELEVANCE_AUDIT.md`
and later change logs.

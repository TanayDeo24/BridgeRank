# Reproducibility

Exact, ordered commands from a clean clone to every number in `docs/BENCHMARK_REPORT.md`.

## 1. Environment

- Python 3.10+ (developed and benchmarked on 3.12.7)
- macOS (Darwin 24.6.0) / 8 logical CPUs — hardware/software for the latency and parallel benchmarks
  is recorded inside `results/latency/*.json` and `results/parallel_experiments/speedup_summary.json`
- No GPU required or used. No paid API. No credentials required for anything in this document
  (`sentence-transformers`/`faiss-cpu`/`mlflow` are open-weight/local; `ANTHROPIC_API_KEY` is
  optional and only used by the pre-existing `support-search failure-report` command, not by
  anything in this reproducibility path)

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate         # .venv\Scripts\Activate.ps1 on Windows
pip install -e ".[dev,ml,dashboard]"
```

The first run of anything that loads `sentence-transformers/all-MiniLM-L6-v2` or
`cross-encoder/ms-marco-MiniLM-L-6-v2` downloads those open-weight models from Hugging Face Hub
(no token required; a `HF_TOKEN` only raises the anonymous rate limit).

## 3. Dataset

Already checked into the repository — no external acquisition step:

- `data/sample_documents.jsonl` / `data/evaluation_queries.jsonl` — the v2, 150/150 corpus used for
  every number in this project.
- `data/sample_documents_v1.jsonl` / `data/evaluation_queries_v1.jsonl` — the original 10/10 corpus,
  preserved unchanged.

To regenerate v2 byte-for-byte (it is deterministic, no randomness):

```bash
PYTHONPATH=src python scripts/generate_corpus.py
```

## 4. Split generation (Gate 3)

```bash
PYTHONPATH=src python scripts/build_splits.py
```

Writes `data/splits.json` (seed=42, deterministic — regenerating reproduces the identical
assignment).

## 5. Lexical-gap definition (Section 17)

```bash
PYTHONPATH=src python scripts/lexical_gap_analysis.py
```

Writes `data/lexical_gap.json`. Deterministic (pure text computation, no model inference).

## 6. Experiment configs

```bash
PYTHONPATH=src python scripts/generate_experiment_configs.py
```

Writes the 14 fixed configs to `configs/experiments/*.json`.

## 7. Train the learned-linear ranking model (dev split only)

```bash
PYTHONPATH=src python scripts/train_learned_model.py
```

Writes `artifacts/models/learned_linear_v1.json` (seed=42, deterministic gradient descent —
identical weights every run).

## 8. Baselines / any single config

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/bm25_only.json --split test
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/hybrid_weighted.json --split test
```

## 9. Validation-only optimization (Sections 18-20)

```bash
PYTHONPATH=src python scripts/optimize.py
```

Writes `results/optimization/search_log.json` and `configs/experiments/final_frozen.json`. **Never
touches the test split.**

## 10. Feature ablations (Section 21, validation split)

```bash
PYTHONPATH=src python scripts/run_ablations.py
```

Writes `results/ablations/*.json`.

## 11. The one final held-out test evaluation (Section 20 step 9)

```bash
PYTHONPATH=src python scripts/final_test_evaluation.py
```

Writes `results/final/*.json` including bootstrap confidence intervals. Deterministic relevance
metrics (verified in this project by running it twice — see `docs/CHANGE_REPORT.md`); only wall-clock
latency numbers vary run to run.

## 12. Latency benchmark (Section 24)

```bash
PYTHONPATH=src python scripts/latency_benchmark.py
```

Writes `results/latency/*.json`. ~15-20 seconds on the reference machine (model loads dominate).

## 13. Parallel experiment benchmark (Sections 22-23)

```bash
PYTHONPATH=src python scripts/parallel_runner.py --manifest configs/manifests/all.json --split val --workers 1 2 4 8
```

Writes `results/parallel_experiments/comparison_workers_*.{json,csv}` and `speedup_summary.json`.
Worker counts should be chosen relative to the actual machine's logical CPU count (printed by the
script) — do not assume more workers is faster without checking (Section 15 of
`docs/BENCHMARK_REPORT.md` shows a machine where 8 workers is *slower* than 4 on an 8-core box).

## 14. Tests

```bash
pytest -q
ruff check src tests loadtest dashboard scripts
pytest --cov=support_search --cov-report=term-missing
```

## 15. MLflow (optional, best-effort)

Any of the commands in steps 8/11 above log to a local MLflow tracking store automatically if
`mlflow` is installed (it is, via the `ml` extra). View with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 16. Expected artifact locations

```
data/splits.json                          Gate 3 split assignment
data/lexical_gap.json                     Section 17 frozen definition + membership
artifacts/models/learned_linear_v1.json   Section 18 trained ranking model
configs/experiments/*.json                15 experiment configs (14 fixed + final_frozen)
results/final/*.json                      the one held-out test run + bootstrap CIs
results/ablations/*.json                  Section 21 ablations
results/optimization/search_log.json      Section 18-20 full search trace
results/latency/*.json                    Section 24 latency benchmark
results/parallel_experiments/*.json,csv   Sections 22-23 parallel benchmark
```

## 17. Random seeds

| Component | Seed | Location |
|---|---|---|
| Split assignment | 42 | `scripts/build_splits.py` |
| Learned-linear model init/training | 42 | `scripts/train_learned_model.py` / `ranking_model.py` |
| Optimization random search (Step 4) | 42 | `scripts/optimize.py` |
| Bootstrap CIs | 42 | `scripts/final_test_evaluation.py` |

## 18. Single orchestration command

There is no single `make benchmark` (the original repository's `Makefile` predates this project and
covers only `pytest`/`ruff`/`build-index`/`evaluate` for the *original* CLI path — extending it was
judged lower-value than keeping the sequence explicit here, since several steps are genuinely
order-dependent one-time operations, not a pipeline meant to be re-run as a black box every time).
Steps 4-13 above, run in that exact order, reproduce every result in `docs/BENCHMARK_REPORT.md`.

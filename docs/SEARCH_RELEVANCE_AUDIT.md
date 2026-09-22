# Search Relevance Foundation Audit

Baseline commit: `93ed1b8`. This audit is based on reading actual source, not README claims.
Every conclusion below cites the file/function inspected.

## 1. Original repository architecture (as executed, not as diagrammed)

Traced call path for a live search request:

```
FastAPI request (api.py:search)
  -> SearchService.search (service.py)
       -> HybridSearchEngine.search (retrieval.py)
            -> BM25Index.scores            [always runs]
            -> DenseIndex.scores           [always runs; FAISS if installed, else numpy dot product]
            -> min-max normalize each, weighted sum by alpha (mode="hybrid")
       -> [only if request asked rerank=true AND settings.reranker_model set]
            -> CrossEncoderReranker.rerank
       -> EventStore.record_query (SQLite/Postgres)
  -> response with per-component scores
```

Traced call path for the benchmark that produced the README numbers:

```
support-search evaluate (cli.py:run_evaluation)
  -> SearchService(settings)                       # builds engine once
  -> for mode in [bm25, dense, hybrid]:
        evaluate(service.engine, queries, top_k=3, mode)   # evaluation.py
             -> engine.search(query, top_k, mode=mode)      # NO rerank argument exists here at all
        -> optional mlflow logging
  -> writes artifacts/evaluation.json
```

**Key structural fact:** `evaluation.py::evaluate()` calls `HybridSearchEngine.search()` directly. It
has no `rerank` parameter and no reference to `CrossEncoderReranker`. The reranker is only reachable
through `SearchService.search(..., rerank=True)`, which is only exercised by the live API
(`api.py`) and by tests, never by the benchmark/evaluation/CLI/quality-gate path.

## 2. README claims vs. actual code — summary

| Claim | Status |
|---|---|
| BM25 lexical baseline | Confirmed, hand-rolled Okapi BM25, not the declared `rank-bm25` dependency (unused) |
| Dense retrieval, configurable ST model | Confirmed; default is `HashingEmbedder`, not a pretrained model |
| Weighted hybrid fusion | Confirmed: min-max normalize, `(1-alpha)*bm25 + alpha*dense`, alpha=0.55 default |
| Cross-encoder reranking "as part of the pipeline" | Code exists, wired to API only, **absent from evaluation** |
| Metrics: P@K, R@K, MRR, nDCG, P50/P95, failed-rate, category slices | Confirmed in `evaluation.py` |
| MLflow logging | Confirmed, best-effort (`try/except ImportError`), only in `cli.py:run_evaluation` |
| "10 graded queries... reported numbers validate the pipeline, not production performance" | Confirmed and independently verified against the data files — this is the single most important caveat in the README and it is accurate |

## 3. Five-gate audit

### Gate 1 — Cross-encoder integration: **NEEDS REPAIR**

- Files/functions inspected: `retrieval.py:CrossEncoderReranker`, `service.py:SearchService.search`,
  `api.py:search`, `evaluation.py:evaluate`, `cli.py:run_evaluation`, `tests/test_api.py`.
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (configurable via `SEARCH_RERANKER_MODEL`), loaded via
  `sentence_transformers.CrossEncoder` — a real, network-dependent, non-deterministic-download model
  requiring the `ml` extra.
- Candidate depth passed to it: `min(settings.candidate_k, len(documents))` = 50 by default, but with
  only 10 documents in the corpus this collapses to "rerank all 10 documents" — the candidate-depth
  parameter is currently unobservable/untestable at meaningful scale.
- Batching: none — `self.model.predict([...])` is called once per request over the full candidate
  list, which is fine at N=10 but would need batching at production candidate depths.
  Failure handling: `SearchService.search` raises `ValueError("Reranking is not configured")` if
  `rerank=True` but no reranker model is set — a deliberate fail-fast, not silent.
- Determinism: cross-encoder inference is deterministic given a fixed model/weights and no dropout at
  eval time (`CrossEncoder.predict` runs the model in eval mode), so this is fine; it was not
  independently re-verified in this session (would require the `ml` extra installed).
- **The decisive finding:** it never participates in the evaluated ranking path that produces
  `artifacts/evaluation.json` or the README benchmark table. It genuinely reorders results at request
  time in the live API, so it is not vaporware, but the project has **zero quantitative evidence** of
  whether reranking helps or hurts — the exact kind of claim the top-level instructions forbid
  ("VALID means it genuinely participates in an evaluated ranking path").
- Consequence: any interview claim of "cross-encoder reranking improves nDCG by X" would currently be
  unsupported by any artifact in this repo.
- Required action: add a `rerank` mode to `evaluate()`/the experiment framework so the cross-encoder is
  benchmarked exactly like `bm25`/`dense`/`hybrid`, with its own latency measurement (cross-encoder
  inference is the most latency-expensive stage and must be measured, not assumed cheap).

### Gate 2 — Evaluation scale and data adequacy: **MISSING** (inadequate for nearly every downstream phase)

Quantified directly from the two JSONL files:

- Total documents: **10**. Total queries: **10** (10 unique — no duplicates). Judged query-document
  pairs: **15** (1–2 relevant docs per query, mean 1.5). Relevance labels used: only grades `1` and `3`
  appear in the data; grade `2` is never used and `0` is implicit (unjudged = non-relevant).
  Query length: 4–8 tokens. Document length: 2–3 sentences (~25–40 tokens of body text). 7 categories,
  1–2 docs/queries each — every category has ≤2 items, so any per-category slice is statistically
  meaningless (n≤2).
- Train/validation/test counts: **0/0/10** — there is no split; the single query file is used for
  every reported metric, for every hyperparameter that exists (BM25 k1/b, alpha=0.55, candidate_k=50),
  and would be used again for any "final" evaluation with zero held-out protection.
- Duplicate-query rate: 0%. Duplicate-document rate: 0%. Near-duplicate rate: not detectable/meaningful
  at n=10 (every document covers a distinct product/category by construction).
- Adequacy determination:
  - **A. Aggregate relevance evaluation:** technically computable, but a 10-query aggregate has such
    wide sampling variance that a ±1-query flip changes MRR by 0.1 and nDCG by a comparable margin.
    Barely adequate as a smoke-test, not as a benchmark.
  - **B. Lexical-gap slicing:** **not adequate.** Any lexical-gap subset carved from 10 queries would
    be a handful of items — not a defensible slice.
  - **C. Feature ablation:** **not adequate** for statistically meaningful deltas; directionally
    informative at best.
  - **D. Parameter tuning:** **not adequate** — tuning alpha/k1/b on 10 queries and reporting the
    result as "optimized" would be curve-fitting to noise.
  - **E. Held-out final comparison:** **not possible** — there is nothing to hold out from without
    destroying the ability to compute anything at all on the remainder.
  - **F. Statistical stability:** **not adequate** — bootstrap CIs on n=10 will be wide enough to be
    non-informative, and must be reported as such rather than hidden.
- The README's own "Limitations and next steps" section already names this exact gap and recommends
  "at least 500 representative queries with independently reviewed graded judgments" — i.e., the
  original author already scoped the correct fix but did not implement it.
- Required action (see escalation note below): expand the corpus and query set within the same
  synthetic technical-support domain, self-authored under the same conventions as the existing 10
  items, large enough to support real splits, ablations, and a lexical-gap slice with double-digit
  query counts per slice — while being explicit in every downstream document that these are
  **author-authored, single-annotator judgments**, not independently reviewed human labels, and that
  this is a deliberate, disclosed limitation rather than a fabricated behavioral signal.

### Gate 3 — Data splits and evaluation leakage: **MISSING**

- No split exists anywhere in the repo (`grep` for "split", "train", "val", "test" in `src/` and
  `data/` returns nothing relevant beyond the `tests/` directory name, which is pytest, not a data
  split).
- Because there is also no tuning history (alpha, k1, b, candidate_k are all just hardcoded defaults,
  never fit), there is currently no *leakage* in the strict sense of "selected using the same queries
  later reported" — there simply was no selection process at all. That is a different problem (no
  valid methodology exists to leak from) but it is equally disqualifying for any claim of "optimized."
- Required action: once the corpus is expanded (Gate 2), create a deterministic, seeded, persisted
  query-level split (dev/validation/test), stratified by category where category counts allow it, and
  route every subsequent tuning/ablation/lexical-gap step through validation only, with test reserved
  for one final pass.

### Gate 4 — Score normalization and fusion: **NEEDS REPAIR**

- `retrieval.py:_normalize` is min-max scaling per query, per component, to `[0,1]`; fusion is
  `(1-alpha)*norm(bm25) + alpha*norm(dense)` with `alpha=0.55` fixed in `config.py` — a **weighted,
  min-max-normalized linear combination**, not raw addition (better than nothing) but the weight
  itself was never selected against data; it is an arbitrary constant.
- No rank-based fusion (e.g., reciprocal rank fusion) or z-score/softmax alternative is implemented or
  compared anywhere in the repo, despite the README table showing hybrid underperforming BM25 on
  MRR/nDCG — exactly the situation where fusion-method choice matters and should be investigated
  rather than assumed.
- Score distributions were not yet measured empirically in this session (requires running the pipeline
  with `pyproject`'s dev/ml extras installed) — this is deferred to the baseline-establishment phase
  and will be measured, not asserted.
- Required action: measure actual BM25/dense/cross-encoder score distributions on the (expanded)
  dataset, compare min-max weighted fusion against at least reciprocal rank fusion on validation only,
  and document the selected method with the validation evidence, per Gate 4's own instructions not to
  implement every method blindly.

### Gate 5 — Relevance judgments: **NEEDS REPAIR**

- Labels are graded 0–3 (per README: 0 irrelevant, 1 marginal, 2 relevant, 3 highly relevant), stored
  directly in `evaluation_queries.jsonl` as `{doc_id: grade}` maps; unjudged documents are implicitly
  treated as grade 0 in every metric in `evaluation.py` (`relevance.get(doc, 0)` in `ndcg_at_k`;
  `precision_at_k`/`recall_at_k` treat anything not in the `relevance` dict's positive-grade set as
  non-relevant).
- With only 10 documents total and 1–2 positively-judged per query, **every** document is in fact
  visible to the (self-)annotator at judgment time — this is a small closed corpus, not a
  pooling/sampling scenario, so the "unjudged = non-relevant" assumption is not hiding a large
  unjudged population; it is, however, still a single-annotator, non-independent judgment process
  (there is no evidence of a second reviewer or inter-annotator agreement check anywhere in the repo).
- Grade `2` is never used in the current 10-query set (only 1 and 3 appear), which slightly reduces the
  practical distinctiveness of the "graded" scale versus a binary one at this sample size, but does not
  invalidate nDCG's mechanics.
- NDCG is mechanically meaningful (graded, correct formula verified by reading `ndcg_at_k`). MRR and
  Precision/Recall are also mechanically correct. All four are *computationally* trustworthy; the open
  issue is evidentiary weight given Gate 2/Gate 5's sample size and annotation independence, not formula
  correctness.
- Required action: when expanding the corpus (Gate 2), keep the same graded 0–3 scale and the same
  explicit definition, author it under one clearly-documented single-annotator process, and state this
  limitation plainly in `BENCHMARK_REPORT.md` and `PROJECT_EVIDENCE.md` rather than presenting the
  judgments as independently validated.

## 4. Identified risks

1. **Headline metrics currently rest on 10 queries.** Any "optimized" or "final" number computed
   without first fixing Gate 2 would be scientifically hollow, no matter how carefully the rest of the
   pipeline is engineered.
2. **Cross-encoder is unevaluated** — a plausible interview question ("does reranking help?") currently
   has no defensible answer in this repo.
3. **Fusion weight is an unexamined constant.** The one number in the README that looks most like a
   "result" (alpha=0.55) is actually just a default nobody tuned.
4. **`rank-bm25` dependency vs. hand-rolled BM25** is a minor but real manifest/code mismatch worth
   fixing for hygiene, not a relevance risk.
5. **ML extras (torch/sentence-transformers/faiss/mlflow) are not installed in the ambient
   environment.** Establishing real dense-retrieval and cross-encoder baselines (as opposed to the
   default hashing embedder) requires installing them, which is a legitimate, bounded, local pip
   install — not a paid API or cloud cost — but is flagged here for transparency since Section 3 of
   the task asks me to flag anything requiring new services/credentials. No credentials are required
   for `sentence-transformers`/`faiss-cpu`/`mlflow` (all open-weight, local, free); `anthropic` is
   already optional and gated behind an API key the user must supply themselves for `failure-report`,
   and nothing in this plan requires it.

## 5. Required repairs before feature expansion (execution order)

1. Expand `data/sample_documents.jsonl` and `data/evaluation_queries.jsonl` within the same domain
   (technical support KB), self-authored, disclosed as such (Gate 2).
2. Create a persisted, seeded, query-level dev/validation/test split over the expanded set (Gate 3).
3. Wire cross-encoder reranking into the evaluation pipeline as a first-class mode, with its own
   latency measurement (Gate 1).
4. Measure real BM25/dense/cross-encoder score distributions and evaluate fusion-method alternatives
   on validation only before freezing one (Gate 4).
5. Document the single-annotator relevance-judgment process explicitly in every results document (Gate 5).

## 6. Decisions requiring escalation

**One decision is flagged for explicit confirmation before proceeding — see the chat message
accompanying this audit.** It concerns *how large* and *by what method* to expand the 10-query/10-document
corpus (Gate 2), since nearly every later phase (splits, ablations, lexical-gap slicing, statistical
stability, parameter tuning) is gated on this choice, and Section 3 of the task's own instructions
requires pausing when "the current repository is so incomplete that a substantial [data] expansion is
required" and when "a required capability cannot be measured honestly with available data." This is
not a request to change the dataset's domain or bring in a third-party dataset — it stays a
self-authored, disclosed, technical-support-KB corpus in the same style as the existing 10 items — but
the scope (how many documents/queries, and the explicit single-annotator-judgment disclosure) is
high-impact enough to confirm before investing the substantial authoring effort it requires.

## 7. Planned next steps (pending the Gate 2 scope confirmation)

1. Author the expanded corpus + queries with disclosed provenance.
2. Build the persisted split.
3. Install `ml`/`dev` extras and establish clean baselines (BM25-only, dense-only, hybrid, cross-encoder
   reranked) on the same split/K/latency protocol.
4. Proceed through Sections 12–41 of the task (feature inventory, diversified ranking-feature
   framework, lexical-gap definition, optimization, ablations, parallel experiment tooling, latency
   benchmarking, tests, MLflow, and the full documentation set) in the order specified.

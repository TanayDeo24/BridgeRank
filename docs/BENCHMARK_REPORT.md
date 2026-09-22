# Benchmark Report

All numbers below are reproducible from the commands in `docs/REPRODUCIBILITY.md` and traceable to a
JSON artifact under `results/`. Nothing here is hand-typed without a source file.

## 1. Evaluation dataset

150 self-authored technical-support KB documents, 150 self-authored graded queries, 10 topic
clusters. See `docs/DATA_PROVENANCE.md` for full detail.

## 2. Dataset provenance

Self-authored, single-annotator, disclosed (not independently reviewed). Domain and judgment
process match the original repository's 10-item set exactly; only scale changed. See
`docs/DATA_PROVENANCE.md`.

## 3. Split methodology

Deterministic, seed=42, query-level, stratified by category: dev 90 / val 30 / test 30 (9/3/3 per
category). Persisted at `data/splits.json`. See `docs/SEARCH_RELEVANCE_AUDIT.md` Gate 3 and
`scripts/build_splits.py`.

## 4. Relevance judgments

Graded 0–3 (0 implicit/irrelevant, 1 marginal, 2 relevant [unused in this dataset], 3 highly
relevant). Single-annotator, disclosed. See `docs/SEARCH_RELEVANCE_AUDIT.md` Gate 5.

## 5. Primary K decision

**Primary K = 5.** The original repository's CLI defaulted to K=3 for its README table but the
engine's own `top_k` default was already 5, and this project's queries average 1.1 relevant
documents each (max 2) — a support-search widget realistically shows a handful of results, and K=5
gives headroom without being so large it stops discriminating. K=1 and K=3 are also computed and
reported below as secondary cuts. This decision was made from corpus/product-use-case reasoning
before any test-split number was inspected (Section 9, Case B → C: the original K=3 was arbitrary/
undocumented, not derived from this project's own analysis, so it was treated as a candidate only).

## 6. Primary metric decision

**Primary metric = nDCG@5.** Relevance is graded (0/1/3), which per Section 10's decision rule
favors nDCG as primary. MRR, Precision@5, Recall@5, and Failed-Query-Rate are all reported as
credible secondary metrics — with only ~1.1 relevant docs/query, MRR is also highly informative and
is reported alongside nDCG throughout, never suppressed when it disagrees.

## 7. Baselines (Section 11)

All baselines below share: same split (test, n=30), same relevance judgments, same K=5, same
latency protocol, `sentence-transformers/all-MiniLM-L6-v2` for every dense/semantic component
(except the disclosed CI-only hashing baseline), fixed model versions, fixed configs
(`configs/experiments/*.json`). Source: `results/final/*.json`.

| Config | nDCG@5 | 95% CI | MRR | P@5 | R@5 | Failed | P95 latency (ms) |
|---|---:|---|---:|---:|---:|---:|---:|
| `bm25_only` | 0.8673 | [0.7462, 0.9667] | 0.8567 | 0.1867 | 0.900 | 10.0% | 12.50 |
| `dense_only` | 0.9991 | [0.9974, 1.0000] | 1.0000 | 0.2067 | 1.000 | 0.0% | 12.61 |
| `hybrid_weighted` (original alpha=0.55, unoptimized) | 0.9587 | [0.9087, 1.0000] | 0.9444 | 0.2067 | 1.000 | 0.0% | 10.40 |
| `cross_encoder_rerank` | 1.0000 | [1.0000, 1.0000] | 1.0000 | 0.2067 | 1.000 | 0.0% | 43.47 |
| `hybrid_learned` (learned-linear fusion) | 0.8831 | [0.7746, 0.9710] | 0.8678 | 0.1933 | 0.933 | 6.7% | 7.23 |

95% CIs are bootstrap (5000 resamples, n=30 queries) — see Section 16 for the caveat this implies.

## 8. Original hybrid results (for comparison to the README's original claim)

The **original repository**, on the **original 10-query v1 set**, reported (unchanged, preserved,
reproducible via `data/*_v1.jsonl`): BM25 nDCG@3=0.725 (best), Hybrid nDCG@3=0.690, Dense
nDCG@3=0.513 — i.e., the original author found BM25 beat hybrid on their 10-query set. On the
**expanded 150-query test split**, at K=5 with a real dense model (not the hashing fallback the
original numbers used), the ordering **reverses**: dense-only and hybrid both clearly beat BM25.
This is not a contradiction — it reflects (a) a real pretrained semantic model replacing the
deterministic hashing embedder, and (b) 30 test queries instead of 10, both legitimate,
disclosed changes in methodology, not a re-analysis of the same data.

## 9. Optimized system (Sections 18-20)

Validation-only grid + random search (`scripts/optimize.py`, `results/optimization/search_log.json`):
best dense weight α=0.70 (vs. original 0.55), weighted_sum beat both RRF and learned_linear on
validation, and **neither cross-encoder reranking nor any of the three new features improved
validation nDCG@5** — both were dropped from the frozen config. Frozen config:
`configs/experiments/final_frozen.json`.

| Config | Test nDCG@5 | 95% CI | MRR | Failed | P95 latency (ms) | Lexical-gap test nDCG@5 (n=5) |
|---|---:|---|---:|---:|---:|---:|
| `final_frozen` | 0.9871 | [0.9619, 1.0000] | 0.9833 | 0.0% | 6.67 | 0.9262 |

**Disclosed tension:** on test, `cross_encoder_rerank` (1.0000) and `dense_only` (0.9991) both
outscore `final_frozen` (0.9871). This is not evidence the validation-based decision was wrong —
retroactively switching after seeing test would be exactly the test-set-tuning this project's
methodology forbids. It is evidence that with n=30 test queries, single-run differences at this
scale (all above 0.98) are close to noise. See Section 16 below.

## 10. Lexical-gap methodology

Frozen definition: `coverage_title < 0.2857` (dev-derived p25 of title-token coverage), chosen over
two alternatives (`jaccard_body`, `coverage_body`) for interpretability and length-robustness. Full
reasoning: `docs/LEXICAL_GAP_METHODOLOGY.md`. 31/150 queries (20.7%) are in the slice: 23 dev / 3 val
/ 5 test.

## 11. Lexical-gap results

| Config | Test lexical-gap nDCG@5 (n=5) | Full test nDCG@5 (n=30) | Gap |
|---|---:|---:|---:|
| `bm25_only` | 0.4036 | 0.8673 | -0.4637 |
| `hybrid_weighted` | 0.8262 | 0.9587 | -0.1325 |
| `hybrid_learned` | 0.4036 | 0.8831 | -0.4795 |
| `cross_encoder_rerank` | 1.0000 | 1.0000 | 0.0000 |
| `final_frozen` | 0.9262 | 0.9871 | -0.0609 |

**n=5 is very small — every number here has enormous sampling variance (a single query changes the
slice's nDCG by ~0.2).** The qualitative pattern (BM25 and the learned-linear fusion both perform
far worse specifically on lexical-gap queries; dense-weighted configs recover most of the gap) is
consistent with the project's central thesis and with the per-query examples in
`docs/FAILURE_ANALYSIS.md`, but is reported as directionally suggestive, not statistically proven,
at this sample size.

## 12. Feature ablations (Section 21, run on validation, `full` architecture)

| Ablation | Val nDCG@5 | Δ vs `full` | P95 latency (ms) |
|---|---:|---:|---:|
| `full` (all signals + reranking) | 0.9655 | — | 44.85 |
| `full_minus_lexical` | 0.9672 | +0.0017 | 41.74 |
| `full_minus_dense` | 0.9655 | +0.0000 | 41.56 |
| **`full_minus_cross_encoder`** | **0.9838** | **+0.0183** | **7.04** |
| `full_minus_title_overlap` | 0.9655 | +0.0000 | 40.43 |
| `full_minus_freshness` | 0.9655 | +0.0000 | 41.27 |
| `full_minus_source_authority` | 0.9655 | +0.0000 | 42.66 |

**The only ablation that improves the system is removing the cross-encoder** — consistent with the
optimization result in Section 9. The three new features (`title_overlap`, `freshness`,
`source_authority`) show exactly zero measured effect in this specific ablation, structurally
explained in `docs/FAILURE_ANALYSIS.md` #7: `full`'s rerank stage gives the cross-encoder full
control of final order (`rerank_weights.cross_encoder=1.0`), so retrieval-stage feature weights only
affect candidate-pool membership, which none of these three features changed at `candidate_k=50` on
this corpus.

## 13. Optimization procedure

Documented fully at the top of `scripts/optimize.py`: fixed search space (alpha grid → fusion-method
comparison → candidate_k/rerank_depth grid → feature-weight random search), fixed objective
(validation nDCG@5), fixed guardrails (P95 latency, lexical-gap-validation non-regression), run once
end-to-end, one stopping rule (drop features/reranking if margin < 0.01), never touched test.

## 14. Latency (Section 24)

Protocol: cold-start timed and reported separately; 15-request warmup; 300 measured single-process
requests per config, cycling the fixed 30-query test set; hardware/software recorded in each
`results/latency/*.json`. Source: `results/latency/all_latency_results.json`.

| Config | Cold start (s) | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---:|---:|---:|---:|
| `bm25_only` | 1.43 | 5.81 | 6.51 | 7.54 |
| `hybrid_weighted` | 1.01 | 5.82 | 6.49 | 9.00 |
| `cross_encoder_rerank` | 2.24 | 33.31 | 37.00 | 44.50 |
| `final_frozen` | 1.18 | 5.87 | 6.57 | 7.12 |

Cross-encoder reranking costs roughly **6x** the P50 latency of every non-reranking configuration —
the concrete production cost behind the decision in Section 9/12 to drop it.

## 15. Parallel experimentation benchmark (Sections 22-23)

14-config manifest, validation split, macOS/8 logical CPUs (`results/parallel_experiments/speedup_summary.json`):

| Workers | Wall clock (s) | Configs/hour | Speedup vs. sequential |
|---:|---:|---:|---:|
| 1 | 20.89 | 2412.93 | 1.00x |
| 2 | 16.50 | 3053.82 | 1.27x |
| 4 | 14.61 | 3450.73 | **1.43x (peak)** |
| 8 | 22.51 | 2238.57 | 0.93x (regression) |

**Workers=4 is the strongest stable configuration on this machine — not the maximum core count.**
Workers=8 performs *worse* than sequential, consistent with oversubscription: each worker process
loads its own copy of the sentence-transformer/torch models, and 8 such processes compete for the
same 8 logical cores' worth of BLAS/torch internal threading. This is reported as the honest,
measured result, not smoothed over — Section 23 explicitly warns against assuming more workers is
always better.

## 16. Statistical stability (Section 26)

- Bootstrap 95% CIs (5000 resamples) are reported for every test-split config in Section 7/9 —
  several near-ceiling configs (`dense_only`, `cross_encoder_rerank`) have CIs of width ≤0.003,
  which reflects that most of their 30 queries score a perfect 1.0, not that the estimate is
  unusually precise in a general sense.
- `bm25_only` and `hybrid_learned` have visibly wider CIs (±0.10-0.12), reflecting genuine query-to-
  query variance in performance.
- **Val→test ordering is not stable at this sample size**: `final_frozen` was the validation winner
  (dropping reranking) but scores below `cross_encoder_rerank`/`dense_only` on test. This is
  reported explicitly as a limitation of n=30 single-run comparisons, not resolved by re-tuning.
- The lexical-gap test slice (n=5) is too small for any confidence-interval claim; Section 11 already
  flags this.

## 17. Negative results (collected, not hidden)

1. Cross-encoder reranking **reduced** validation nDCG@5 (0.9847→0.9655) while multiplying P95
   latency ~6-9x (Sections 9, 12, 14).
2. The machine-learned (`learned_linear`) fusion strategy **underperformed** hand-tuned weighted-sum
   on both validation (0.9477 vs 0.9847) and test (0.8831 vs 0.9871).
3. Two of the two new ranking features (`freshness`, `source_authority`) showed **zero measured
   ranking effect** in ablation, and the learned model assigned them **negative** coefficients.
4. Hybrid fusion **never** strictly beat both BM25 and dense simultaneously on any of the 30 test
   queries (`docs/FAILURE_ANALYSIS.md` #3) — its aggregate advantage comes from different queries
   being rescued on different components, not from consistent per-query synergy.
5. Parallel speedup **regressed below sequential** at 8 workers on an 8-core machine.

## 18. Failure analysis summary

See `docs/FAILURE_ANALYSIS.md` for the full, sourced, 10-category breakdown with concrete query
examples for 9/10 categories (2 categories explicitly disclosed as "not observed at this corpus
scale" rather than fabricated).

## 19. Limitations

- 150 documents / 150 queries is enough for real splits and a lexical-gap slice, but small enough
  that many configurations reach near-ceiling performance (nDCG@5 > 0.95) — this benchmark measures
  *relative* engineering choices (which fusion, whether reranking helps) more reliably than it
  measures *absolute* production-scale search quality.
- Single-annotator, self-authored relevance judgments (disclosed throughout, `docs/DATA_PROVENANCE.md`).
- Val (n=30) and test (n=30) splits are large enough for a rough signal but small enough that
  single-run comparisons near the ceiling are noisy (Section 16).
- Lexical-gap val (n=3) / test (n=5) slices are too small for confidence-interval claims; reported
  directionally with the caveat stated every time.
- The learned-linear model is trained on a highly imbalanced binary-relevance signal without
  negative sampling or a pairwise/listwise loss — a known simplification, not a bug.

# Failure Analysis

All examples below are pulled directly from saved per-query rows in `results/final/*.json` and
`results/ablations/*.json` (test and validation splits respectively) — none are invented. Where a
category has no real example at this corpus scale, that is stated explicitly rather than filled in.

## 1. BM25 beats dense

**1/30 test queries.** `"home printer unreachable while vpn is on"` — BM25 nDCG@5 exceeds dense's.
The query shares literal vocabulary ("vpn", "printer" is absent from the target doc, but
"reach"/"vpn" terms overlap directly with the target document
*"Split-tunnel VPN cannot reach the printer on the home network"*), which is exactly the profile
where lexical matching is expected to do well and does.

## 2. Dense beats BM25

**3/30 test queries**, all involving genuine paraphrase (title-token coverage below the frozen
lexical-gap threshold): `"wifi disconnects on just my laptop"` (target title: *"Wi-Fi keeps
disconnecting on one laptop"*), `"vpn app won't connect at all"` (target: *"VPN client will not
establish a tunnel"*), `"uninstall didn't fully remove the app, reinstall fails"` (target:
*"Uninstalling an application leaves behind leftover files that block reinstall"*). BM25 either
missed the target entirely (failed query) or ranked it below a lexically-closer but wrong document
(e.g., it matched `kb-net-013`/`kb-net-006` on more surface overlap than the true target). This is
the semantic-retrieval value proposition working as expected — the reason the project exists.

## 3. Hybrid beats both BM25 and dense simultaneously

**0/30 test queries.** At this corpus size and difficulty, dense-only already reaches nDCG@5=0.999
on test (near-ceiling), leaving essentially no room for hybrid fusion to beat *both* individual
retrievers on the same query — whichever one already gets a query right, the other typically does
too. This is disclosed as a genuine limitation of benchmarking hybrid fusion's *complementary* value
on a corpus this small and this close to ceiling, not evidence that fusion has no value in general.

## 4. Hybrid hurts (underperforms the better of BM25/dense)

**3/30 test queries** — the same three queries as category 2. Hybrid's fixed 0.45/0.55 weighting
(the *un-optimized*, original-repository default `hybrid_weighted` config) dilutes dense's correct
signal with BM25's wrong one on exactly the paraphrase queries where BM25 fails outright. This is
precisely why Section 18's validation-only optimization pushed the dense weight up to 0.70 in
`final_frozen` — and why `final_frozen` had **zero** failed test queries where `hybrid_weighted`
(unoptimized alpha=0.55) had none failed but still underperformed on these three relative to
dense-only.

## 5. Cross-encoder fixes ranking

**3/30 test queries** — again the same three paraphrase queries: reranking recovers all three to
nDCG@5=1.0 (`cross_encoder_rerank` scored a perfect 1.0000 on the full test split). The cross-encoder
genuinely helps exactly where lexical matching fails and the retrieval-stage fusion weight was not
yet optimized toward dense.

## 6. Cross-encoder hurts ranking

**Concrete example found on the validation split** (`results/ablations/full.json` vs
`full_minus_cross_encoder.json`): for `"can't remember my login password"`, the retrieval stage
alone correctly ranks `kb-acc-001` ("Reset a forgotten account password") first (nDCG@5=1.0), but
adding cross-encoder reranking demotes it in favor of `kb-sof-013` (nDCG@5=0.387) — a keyword-driven
false positive on the word "password" in an unrelated software document. This is the single ablation
example (out of 90 validation queries) where reranking made things strictly worse, and it is the
concrete instance behind the aggregate validation finding that cross-encoder reranking reduced
nDCG@5 from 0.9847 to 0.9655 on validation (Section optimize.py, Step 3).

**Important disclosed tension:** on the *test* split, cross-encoder reranking scored a perfect 1.0000
— higher than the validation-selected `final_frozen` config (0.9871) which dropped reranking. This
is not evidence the validation-based decision was "wrong": with only 30 test queries, a single-run
comparison is noisy (see the bootstrap CIs in `results/final/all_final_results.json`), and retroactively
switching back to reranking after seeing the test number would be exactly the test-set-tuning this
project's methodology forbids. The frozen decision stands; the discrepancy itself is reported as a
finding about small-sample statistical stability (Section 26), not resolved by re-tuning.

## 7. New feature (title_overlap / freshness / source_authority) improves ordering

**No example found.** The ablation study (`results/ablations/ablation_summary.json`) shows a
measured delta of 0.0000 nDCG@5 for removing any of `title_overlap`, `freshness`, or
`source_authority` from the `full` config on validation. The reason, diagnosed from the config
itself: `full`'s `rerank_weights` gives the cross-encoder a weight of 1.0 and the retrieval-stage
fused score a weight of 0.0 — so whenever reranking is active (as it is in `full` and all of its
single-feature ablations), the retrieval-stage feature weights only affect **which documents enter
the rerank candidate pool** (top `candidate_k=50` of 150), not the final order. None of these three
low-weight features (0.05–0.10) changed pool membership for any validation query at this corpus
size. This is reported as a genuine negative/null result, not hidden.

## 8. New feature hurts ranking

**No ranking-level example** for the reason given in #7 (the features' effect was structurally
masked by reranking). However, the **learned-linear model** (`ranking_model.py`, trained on dev)
did assign `freshness` and `source_authority` **negative** coefficients (`-0.659` and `-0.460`
respectively; see `artifacts/models/learned_linear_v1.json`), meaning the logistic classifier found
these two features **anti-correlated** with relevance in the training data — expected, since both
were authored with plausible-looking but relevance-independent values (Section 13's own caveat:
they are legitimate document attributes, not proven-predictive ones on this corpus). The
`hybrid_learned` fusion strategy scored lower than hand-tuned weighted-sum on both validation
(0.9477 vs 0.9847) and test (0.8831 vs 0.9871) — a disclosed negative result for the
machine-learned fusion approach on a dataset this small (13,500 training rows but only ~1% positive
rate, i.e. ~94 positive examples, is a thin training signal for 5 coefficients).

## 9. Lexical-gap query remains unsolved

**1/5 test lexical-gap queries** (`data/lexical_gap.json`): `"vpn app won't connect at all"` scores
nDCG@5=0.6309 under the frozen `final_frozen` configuration — the target document (*"VPN client
will not establish a tunnel"*) is retrieved but not ranked first; the very small candidate corpus
(only 150 documents, 15 of them in the "network" category alone, several also VPN-related) makes
this a genuinely hard disambiguation even after paraphrase-aware dense retrieval. This is the
project's clearest remaining, disclosed unsolved case.

## 10. Final system still fails badly

**No example on the 30-query test split** — `final_frozen` had 0 failed queries (a document with
nonzero relevance was retrieved in the top 5 for all 30). This is disclosed as a limitation of
corpus scale and difficulty (Gate 2 discussion in `docs/SEARCH_RELEVANCE_AUDIT.md`) rather than
claimed as evidence of a solved problem: at 150 documents, most queries have exactly one plausible
answer and the retrieval task is closer to disambiguation among a handful of topically related
documents than to open-domain search. A production-scale corpus (thousands of documents, more
topic overlap, noisier real queries) would very likely surface hard failures this benchmark cannot.

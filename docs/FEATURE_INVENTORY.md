# Feature Inventory (Section 12)

Every candidate ranking signal considered for this project, whether implemented or not, with the
reasoning for the decision. "Implemented" features live in `src/support_search/features.py` and are
registered in `FEATURE_REGISTRY`.

| Candidate | Source | Type | Coverage | Query-dependent? | Doc-dependent? | Static/Dynamic | Available at ranking time? | Leakage risk | Runtime cost | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| **bm25** | Computed from corpus text | float | 100% | Yes | Yes | Dynamic (per-query) | Yes | None | Cheap (O(corpus)) | **Implemented** — original repository capability, kept |
| **dense** | Sentence-Transformer / hashing embedding cosine similarity | float | 100% | Yes | Yes | Dynamic (per-query) | Yes | None | Moderate (one encode call + dot products) | **Implemented** — original capability, kept |
| **cross_encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` pairwise relevance score | float | 100% of rerank pool | Yes | Yes | Dynamic (per-query, per-candidate) | Yes | None | Expensive (transformer forward pass per candidate) | **Implemented** — Gate 1 repair wired it into evaluation |
| **title_overlap** | Query tokens ∩ document title tokens | float [0,1] | 100% | Yes | Yes | Dynamic | Yes | None | Cheap (set intersection) | **Implemented** — cheap, interpretable, genuinely distinct computation from BM25 |
| **freshness** | `metadata.last_updated` exponential decay | float [0,1] | 100% (all 150 docs carry this field) | No | Yes | Static per document | Yes | None (document-only) | Cheap | **Implemented** — Section 13 candidate A |
| **source_authority** | `metadata.source_authority` (official/community) | float {0.5, 1.0} | 100% | No | Yes | Static per document | Yes | None | Cheap | **Implemented** — Section 13 candidate B |
| category/product exact-match boost | Existing `Document.category`/`.product` fields | categorical | 100% | Yes (if filters used) | Yes | Static | Yes | None | Cheap | **Not implemented as a ranking feature** — already used as a hard *filter* in `HybridSearchEngine.search`, not a ranking signal; turning it into a soft ranking boost was considered but rejected as redundant with the hard filter already in production use |
| document length | `len(searchable_text)` | int | 100% | No | Yes | Static | Yes | None | Cheap | **Not implemented** — no defensible relevance rationale on a corpus where every document is already a similarly-scoped, focused troubleshooting article (body length range is narrow: 2–4 sentences); would be noise, not signal |
| difficulty (`metadata.difficulty`) | Existing field: basic/intermediate/advanced | categorical | 100% | No | Yes | Static | Yes | None | Cheap | **Not implemented** — no principled reason a "basic" article should rank above/below an "advanced" one for a matching query; using it would be an arbitrary, unjustified boost |
| click-through rate | N/A | — | 0% | — | — | — | — | — | — | **Explicitly excluded** — no real user click/query-log data exists anywhere in this repository; fabricating one would violate the project's core constraint against invented behavioral signals (Section 1/Section 37 of the task) |
| popularity / view count | N/A | — | 0% | — | — | — | — | — | — | **Explicitly excluded** — same reason as CTR: no genuine data exists |
| personalization / user history | N/A | — | 0% | — | — | — | — | — | — | **Explicitly excluded** — out of this project's scope by the task's own instructions (query-log/behavioral work belongs to a separate project) |
| entity/product exact match (e.g. "VPN" named-entity extraction) | Would require an NER step | float | N/A | Yes | Yes | Dynamic | Yes, if built | Low | Moderate (NER inference) | **Considered, not implemented** — `title_overlap` already captures most of this value at near-zero cost on a 150-document corpus; a dedicated NER pipeline was judged disproportionate complexity for the marginal expected gain (Section 40: do not overbuild) |
| synonym/thesaurus expansion score | Would require a synonym dictionary or WordNet | float | N/A | Yes | Yes | Dynamic | Yes, if built | Low | Moderate | **Considered, not implemented** — the dense embedding feature already exists specifically to capture synonym/paraphrase relationships; adding a second, weaker mechanism for the same purpose was judged redundant |

## Selected new feature families (Section 13)

Ranked by relevance rationale, coverage, leakage risk, runtime cost, complementarity, interpretability,
and availability at ranking time — the two chosen were:

1. **`freshness`** — every document has a genuine authored `last_updated` date; recency is a
   standard, well-understood support-KB relevance signal (README's own "Limitations and next
   steps" section did not mention this, but it is a natural, low-risk, zero-behavioral-data
   addition). Zero leakage risk (static document property), zero query dependency, negligible cost.
2. **`source_authority`** — every document has a genuine authored `official`/`community` tag,
   analogous to a real support-KB's distinction between an official runbook and a community-wiki
   tip. Same low-risk profile as freshness.

Both were implemented, both were then honestly evaluated in the ablation study
(`docs/BENCHMARK_REPORT.md` Section 12 / `docs/FAILURE_ANALYSIS.md` #7-#8) and found to have
**no measurable effect** on this corpus's ranking quality — a disclosed negative result, not
suppressed because it does not fit a nicer story.

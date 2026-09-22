# Model card: Support Search v0.1

## Intended use

Retrieve English technical-support articles for employee and customer support queries. It is a decision-support
tool, not an autonomous diagnostic system.

## System

The service combines BM25 lexical scores and normalized dense cosine similarity. The lightweight default uses
deterministic hashed token and bigram features; the production profile uses a Sentence Transformers bi-encoder
and optionally a cross-encoder reranker. Metadata filters constrain category and product.

## Evaluation

The checked-in relevance set is deliberately small and exists to verify the pipeline. Before deployment, label
at least 500 representative queries with graded relevance, report Precision@K, Recall@K, MRR, nDCG, P50/P95
latency, failed-query rate, category slices, confidence intervals, and BM25/dense/hybrid/reranking ablations.

## Limitations and risks

Quality degrades for new products, non-English queries, ambiguous acronyms, and stale articles. Retrieval logs
may contain personal data; redact secrets, apply access controls and retention limits, and never log raw health
or authentication data. Human review is required for destructive or security-sensitive remedies.

## Monitoring

Alert on latency SLO violations, error and empty-result rates, embedding/input drift, category-level nDCG drops,
and index/source freshness. Re-evaluate and version the index whenever documents or embedding models change.


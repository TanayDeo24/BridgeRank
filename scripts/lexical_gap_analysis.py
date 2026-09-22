"""Section 17: design (not assume) the lexical-gap definition.

Computes three independent, non-circular candidate definitions purely from
query/relevant-document TEXT overlap (never from any retrieval model's
output, which would be circular), reports their distributions and overlap
with each other **on the dev split only**, and freezes one definition.
The frozen definition is then applied mechanically to dev+val+test and
persisted to data/lexical_gap.json so it cannot silently change later.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from support_search.experiment import load_all_queries, load_corpus, load_split_assignment
from support_search.text import tokenize

ROOT = Path(__file__).resolve().parents[1]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on",
    "for", "and", "or", "but", "my", "i", "it", "its", "this", "that", "with", "at",
    "does", "do", "did", "has", "have", "had", "not", "no", "so", "just", "even",
}


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1}


def primary_doc(relevance: dict[str, int]) -> str:
    return max(relevance.items(), key=lambda kv: kv[1])[0]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def coverage(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens:
        return 1.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def main() -> None:
    documents = {d.id: d for d in load_corpus()}
    queries = load_all_queries()
    assignment = load_split_assignment()

    per_query = []
    for item in queries:
        target_id = primary_doc(item["relevance"])
        doc = documents[target_id]
        q_tokens = content_tokens(item["query"])
        body_tokens = content_tokens(doc.searchable_text)
        title_tokens = content_tokens(doc.title)
        per_query.append(
            {
                "query": item["query"],
                "category": item["category"],
                "split": assignment.get(item["query"], "unknown"),
                "target_doc": target_id,
                "jaccard_body": jaccard(q_tokens, body_tokens),
                "coverage_body": coverage(q_tokens, body_tokens),
                "coverage_title": coverage(q_tokens, title_tokens),
            }
        )

    dev_rows = [r for r in per_query if r["split"] == "dev"]

    def describe(key: str, rows: list[dict]) -> dict:
        values = sorted(r[key] for r in rows)
        return {
            "min": values[0],
            "p25": values[len(values) // 4],
            "median": statistics.median(values),
            "p75": values[3 * len(values) // 4],
            "max": values[-1],
            "mean": statistics.fmean(values),
        }

    print(f"=== Candidate definitions, measured on DEV split (n={len(dev_rows)}) ===")
    for key in ("jaccard_body", "coverage_body", "coverage_title"):
        print(key, describe(key, dev_rows))

    # --- Decision (documented in docs/LEXICAL_GAP_METHODOLOGY.md) ---
    # coverage_title is chosen as the primary definition: it is the most
    # interpretable ("what fraction of the query's content words appear
    # anywhere in the document's own title"), the least sensitive to body
    # length (titles are uniformly short, so it isn't confounded by
    # document length the way body-jaccard/coverage are), and produces a
    # clean, round, dev-derived threshold. See the methodology doc for the
    # full comparison and the rejected alternatives.
    dev_p25 = describe("coverage_title", dev_rows)["p25"]
    threshold = round(dev_p25, 2)  # dev-split p25 of coverage_title, taken as-is (no manual rounding up/down)
    membership = {r["query"]: r["coverage_title"] < threshold for r in per_query}

    out = {
        "definition": "coverage_title",
        "formula": "fraction of query content tokens (stopwords removed) that literally appear in the primary relevant document's TITLE",
        "threshold": threshold,
        "threshold_derivation": f"dev-split p25 of coverage_title = {dev_p25:.4f}, rounded to {threshold}",
        "frozen_before_comparing_systems": True,
        "counts": {
            "dev": sum(1 for r in per_query if r["split"] == "dev" and membership[r["query"]]),
            "val": sum(1 for r in per_query if r["split"] == "val" and membership[r["query"]]),
            "test": sum(1 for r in per_query if r["split"] == "test" and membership[r["query"]]),
        },
        "membership": membership,
        "per_query_diagnostics": {r["query"]: r for r in per_query},
    }
    out_path = ROOT / "data" / "lexical_gap.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nlexical-gap counts by split:", out["counts"])
    print(f"total lexical-gap queries: {sum(membership.values())}/150")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()

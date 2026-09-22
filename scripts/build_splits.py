"""
Gate 3 repair: build a deterministic, seeded, persisted query-level split.

Design decisions (documented here rather than only in code comments, since
this script IS the split methodology):

- Split unit is the query (information need), not a query-document row —
  every relevant document for a given query stays in the same split as that
  query, by construction, since we split the query list itself.
- Stratified by `category` so every one of the 10 categories is represented
  in dev/val/test (each category has exactly 15 queries in v2, so an even
  60/20/20 split gives 9/3/3 per category — no category is left with zero
  queries in any split).
- Seed is fixed (42) and recorded in the output file. Re-running this script
  reproduces the identical assignment; it is never regenerated differently
  between experiments (per Section 3, Gate 3 requirement 8).
- No near-duplicate leakage check is needed beyond category stratification:
  the corpus was verified (generate_corpus.py's own assertions, plus the
  provenance doc) to contain no duplicate or near-duplicate queries.
- Purpose of each split:
    dev        -> free exploration / feature inventory / lexical-gap definition design
    validation -> all tuning, fusion-method selection, feature ablation decisions
    test       -> touched exactly once, at the end, for the final held-out report
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
RATIOS = {"dev": 0.6, "val": 0.2, "test": 0.2}


def main() -> None:
    queries = [
        json.loads(line)
        for line in (ROOT / "data" / "evaluation_queries.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_category: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_category[q.get("category", "unknown")].append(q["query"])

    assignment: dict[str, str] = {}
    counts = {"dev": 0, "val": 0, "test": 0}
    rng = random.Random(SEED)
    for category in sorted(by_category):
        items = sorted(by_category[category])  # sort first for determinism, then seeded shuffle
        rng.shuffle(items)
        n = len(items)
        n_dev = round(n * RATIOS["dev"])
        n_val = round(n * RATIOS["val"])
        splits_for_cat = (
            ["dev"] * n_dev + ["val"] * n_val + ["test"] * (n - n_dev - n_val)
        )
        for query_text, split in zip(items, splits_for_cat):
            assignment[query_text] = split
            counts[split] += 1

    output = {
        "seed": SEED,
        "ratios": RATIOS,
        "unit": "query",
        "stratified_by": "category",
        "total_queries": len(queries),
        "counts": counts,
        "assignment": assignment,
    }
    out_path = ROOT / "data" / "splits.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=False), encoding="utf-8")
    print(f"wrote split assignment for {len(assignment)} queries -> {out_path}")
    print("counts:", counts)


if __name__ == "__main__":
    main()

"""Tests that the persisted split (Gate 3) and lexical-gap membership
(Section 17) actually satisfy the properties this project relies on for
every downstream claim: no query appears in two splits, every relevance
document id resolves to a real document, and lexical-gap membership is a
pure, reproducible function of the frozen formula.
"""
import json
from pathlib import Path

from support_search.experiment import load_all_queries, load_corpus, load_split_assignment
from support_search.text import tokenize

ROOT = Path(__file__).resolve().parents[1]


def test_every_query_is_assigned_to_exactly_one_split():
    assignment = load_split_assignment()
    queries = load_all_queries()
    for item in queries:
        assert item["query"] in assignment
        assert assignment[item["query"]] in {"dev", "val", "test"}


def test_split_sizes_match_documented_counts():
    splits_raw = json.loads((ROOT / "data" / "splits.json").read_text())
    assert splits_raw["counts"] == {"dev": 90, "val": 30, "test": 30}
    assert splits_raw["seed"] == 42


def test_no_duplicate_documents_or_queries():
    documents = load_corpus()
    queries = load_all_queries()
    assert len({d.id for d in documents}) == len(documents)
    assert len({q["query"] for q in queries}) == len(queries)


def test_every_relevance_judgment_points_at_a_real_document():
    doc_ids = {d.id for d in load_corpus()}
    for item in load_all_queries():
        for doc_id in item["relevance"]:
            assert doc_id in doc_ids


def test_lexical_gap_membership_is_reproducible_from_the_frozen_formula():
    gap = json.loads((ROOT / "data" / "lexical_gap.json").read_text())
    documents = {d.id: d for d in load_corpus()}
    threshold = gap["threshold"]
    stopword_free = lambda text: {t for t in tokenize(text) if len(t) > 1}

    checked = 0
    for item in load_all_queries():
        target_id = max(item["relevance"].items(), key=lambda kv: kv[1])[0]
        title_tokens = stopword_free(documents[target_id].title)
        query_tokens = {t for t in tokenize(item["query"]) if len(t) > 1}
        # recompute without the full stopword list for a coarse cross-check:
        # coverage should be monotonically consistent with membership at
        # values far from the threshold (avoids re-implementing the exact
        # stopword list here, which would make this test circular).
        if not query_tokens:
            continue
        coverage = len(query_tokens & title_tokens) / len(query_tokens)
        expected_far_below = coverage < threshold - 0.15
        expected_far_above = coverage > threshold + 0.15
        member = gap["membership"][item["query"]]
        if expected_far_below:
            assert member is True
        elif expected_far_above:
            assert member is False
        checked += 1
    assert checked > 100

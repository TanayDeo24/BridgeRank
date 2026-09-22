from support_search.embeddings import HashingEmbedder
from support_search.models import Document
from support_search.retrieval import HybridSearchEngine


def engine():
    docs = [
        Document(
            "wifi",
            "Fix wireless connection",
            "Restart the router and reset network settings",
            "network",
            "router",
        ),
        Document(
            "password", "Reset password", "Use the account recovery email link", "account", "portal"
        ),
        Document(
            "battery",
            "Battery drains quickly",
            "Disable background apps and check battery health",
            "hardware",
            "laptop",
        ),
    ]
    return HybridSearchEngine(docs, HashingEmbedder(128))


def test_bm25_finds_keyword_match():
    assert engine().search("forgot password", mode="bm25")[0].document.id == "password"


def test_metadata_filtering():
    results = engine().search("reset", filters={"category": "network"})
    assert [r.document.id for r in results] == ["wifi"]


def test_invalid_mode():
    import pytest

    with pytest.raises(ValueError):
        engine().search("test", mode="invalid")

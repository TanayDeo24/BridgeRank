import json

import pytest

from support_search.artifacts import build_artifact, load_artifact
from support_search.embeddings import HashingEmbedder
from support_search.io import load_documents


def test_artifact_round_trip_and_integrity(tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text(
        json.dumps({"id": "1", "title": "Reset VPN", "text": "Reconnect client"}) + "\n",
        encoding="utf-8",
    )
    documents = load_documents(source)
    embedder = HashingEmbedder(32)
    output = tmp_path / "index" / "current"
    manifest = build_artifact(documents, embedder, source, output)
    loaded, vectors, loaded_manifest = load_artifact(output, embedder.name)
    assert loaded == documents
    assert vectors.shape == (1, 32)
    assert loaded_manifest["source_sha256"] == manifest["source_sha256"]

    with (output / "documents.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        load_artifact(output, embedder.name)

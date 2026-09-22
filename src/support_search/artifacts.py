from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .models import Document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_artifact(
    documents: list[Document], embedder: Embedder, source: Path, output: Path
) -> dict:
    """Write an immutable, checksummed dense-index bundle via atomic directory promotion."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp)
        docs_path, vectors_path = staging / "documents.jsonl", staging / "vectors.npy"
        docs_path.write_text(
            "".join(json.dumps(d.__dict__, sort_keys=True) + "\n" for d in documents),
            encoding="utf-8",
        )
        vectors = embedder.encode([d.searchable_text for d in documents])
        np.save(vectors_path, vectors, allow_pickle=False)
        manifest = {
            "schema_version": 1,
            "created_at": int(time.time()),
            "source": str(source.resolve()),
            "source_sha256": _sha256(source),
            "document_count": len(documents),
            "embedding_model": embedder.name,
            "dimension": int(vectors.shape[1]),
            "python": platform.python_version(),
            "files": {"documents.jsonl": _sha256(docs_path), "vectors.npy": _sha256(vectors_path)},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if output.exists():
            backup = output.with_name(output.name + ".previous")
            if backup.exists():
                for child in backup.iterdir():
                    child.unlink()
                backup.rmdir()
            os.replace(output, backup)
        os.replace(staging, output)
        return manifest


def load_artifact(path: Path, expected_model: str) -> tuple[list[Document], np.ndarray, dict]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["embedding_model"] != expected_model:
        raise RuntimeError("Artifact embedding model does not match configured model")
    for filename, checksum in manifest["files"].items():
        if _sha256(path / filename) != checksum:
            raise RuntimeError(f"Artifact checksum failed: {filename}")
    documents = [
        Document(**json.loads(line))
        for line in (path / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    vectors = np.load(path / "vectors.npy", allow_pickle=False)
    if len(documents) != len(vectors) or len(documents) != manifest["document_count"]:
        raise RuntimeError("Artifact document/vector count mismatch")
    return documents, vectors.astype(np.float32), manifest

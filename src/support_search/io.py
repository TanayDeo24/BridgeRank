from __future__ import annotations

import json
from pathlib import Path

from .models import Document


def load_documents(path: str | Path) -> list[Document]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Document(**json.loads(line)) for line in handle if line.strip()]


def load_queries(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

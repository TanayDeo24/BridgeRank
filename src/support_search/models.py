from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    category: str = "general"
    product: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    rank: int
    retrieval_scores: dict[str, float] = field(default_factory=dict)

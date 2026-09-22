from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    insert,
    select,
)
from sqlalchemy.exc import IntegrityError

metadata = MetaData()
queries = Table(
    "queries",
    metadata,
    Column("query_id", String(36), primary_key=True),
    Column("created_at", Float, nullable=False),
    Column("query_hash", String(64), nullable=False),
    Column("query_text", Text),
    Column("mode", String(16), nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("result_ids", Text, nullable=False),
    Column("failed", Boolean, nullable=False),
    Column("event_metadata", Text, nullable=False),
)
Index("ix_queries_created_at", queries.c.created_at)
feedback = Table(
    "feedback",
    metadata,
    Column("feedback_id", String(36), primary_key=True),
    Column("query_id", String(36), ForeignKey("queries.query_id"), nullable=False),
    Column("document_id", String(200), nullable=False),
    Column("relevance", Integer, nullable=False),
    Column("created_at", Float, nullable=False),
    CheckConstraint("relevance BETWEEN 0 AND 3", name="ck_feedback_relevance"),
)
Index("ix_feedback_query_id", feedback.c.query_id)


class EventStore:
    def __init__(self, target: Path | str):
        if isinstance(target, Path):
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{target.resolve().as_posix()}"
        else:
            url = target
        connect_args = (
            {"timeout": 10, "check_same_thread": False} if url.startswith("sqlite") else {}
        )
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._configure_sqlite)
        metadata.create_all(self.engine)

    @staticmethod
    def _configure_sqlite(connection, _record) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def record_query(self, item: dict[str, Any], retain_text: bool = False) -> None:
        with self.engine.begin() as db:
            db.execute(
                insert(queries).values(
                    query_id=item["query_id"],
                    created_at=time.time(),
                    query_hash=item["query_hash"],
                    query_text=item["query"] if retain_text else None,
                    mode=item["mode"],
                    latency_ms=item["latency_ms"],
                    result_ids=json.dumps(item["result_ids"]),
                    failed=item["failed"],
                    event_metadata=json.dumps(item["metadata"]),
                )
            )

    def record_feedback(
        self, feedback_id: str, query_id: str, document_id: str, relevance: int
    ) -> None:
        try:
            with self.engine.begin() as db:
                exists = db.execute(
                    select(queries.c.query_id).where(queries.c.query_id == query_id)
                ).first()
                if not exists:
                    raise KeyError(query_id)
                db.execute(
                    insert(feedback).values(
                        feedback_id=feedback_id,
                        query_id=query_id,
                        document_id=document_id,
                        relevance=relevance,
                        created_at=time.time(),
                    )
                )
        except IntegrityError as exc:
            raise ValueError("Invalid or duplicate feedback") from exc

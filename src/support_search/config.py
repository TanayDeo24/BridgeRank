from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SEARCH_", extra="ignore")

    data_path: Path = Path("data/sample_documents.jsonl")
    artifact_dir: Path = Path("artifacts/index/current")
    database_path: Path = Path("artifacts/search.db")
    database_url: str | None = None
    embedding_model: str = "hashing"
    reranker_model: str | None = None
    index_type: str = "flat"
    hybrid_alpha: float = Field(default=0.55, ge=0, le=1)
    candidate_k: int = Field(default=50, ge=1, le=1000)
    max_top_k: int = Field(default=50, ge=1, le=100)
    api_key: str | None = None
    retain_query_text: bool = False
    environment: str = "development"

    @field_validator("index_type")
    @classmethod
    def valid_index_type(cls, value: str) -> str:
        if value not in {"flat", "hnsw"}:
            raise ValueError("index_type must be flat or hnsw")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

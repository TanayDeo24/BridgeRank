from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from .config import get_settings
from .service import SearchService

logger = logging.getLogger("support_search")
REQUESTS = Counter("search_requests_total", "Search requests", ["mode", "status"])
LATENCY = Histogram("search_latency_seconds", "Search request latency", ["mode", "reranked"])
FAILED = Counter("search_failed_queries_total", "Queries returning no results", ["mode"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.search = await asyncio.to_thread(SearchService, settings)
    app.state.ready = True
    yield
    app.state.ready = False


app = FastAPI(title="Support Search API", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))[:128]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled_request_error", extra={"request_id": request_id})
        response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{(time.perf_counter() - started) * 1000:.3f}"
    return response


def authorize(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = request.app.state.settings.api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


class ResultResponse(BaseModel):
    id: str
    title: str
    text: str
    category: str
    product: str
    score: float
    rank: int
    scores: dict[str, float]


class SearchResponse(BaseModel):
    query_id: str
    query: str
    mode: str
    reranked: bool
    latency_ms: float
    results: list[ResultResponse]


class FeedbackRequest(BaseModel):
    query_id: str
    document_id: str = Field(min_length=1, max_length=200)
    relevance: int = Field(ge=0, le=3)


@app.get("/health/live")
async def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Service is not ready")
    service = request.app.state.search
    return {
        "status": "ready",
        "documents": len(service.documents),
        "embedding_model": service.embedder.name,
        "artifact": service.manifest.get("source_sha256"),
    }


@app.get("/health", include_in_schema=False)
async def health(request: Request):
    return await readiness(request)


@app.get("/v1/search", response_model=SearchResponse, dependencies=[Depends(authorize)])
async def search(
    request: Request,
    q: Annotated[str, Query(min_length=2, max_length=500)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 5,
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid",
    rerank: bool = False,
    category: str | None = None,
    product: str | None = None,
):
    if top_k > request.app.state.settings.max_top_k:
        raise HTTPException(status_code=422, detail="top_k exceeds configured maximum")
    filters = {k: v for k, v in {"category": category, "product": product}.items() if v}
    try:
        with LATENCY.labels(mode, str(rerank).lower()).time():
            query_id, results, latency = await asyncio.to_thread(
                request.app.state.search.search, q, top_k, filters, mode, rerank
            )
        REQUESTS.labels(mode, "success").inc()
        if not results:
            FAILED.labels(mode).inc()
        return SearchResponse(
            query_id=query_id,
            query=q,
            mode=mode,
            reranked=rerank,
            latency_ms=round(latency, 3),
            results=[
                ResultResponse(
                    id=r.document.id,
                    title=r.document.title,
                    text=r.document.text,
                    category=r.document.category,
                    product=r.document.product,
                    score=r.score,
                    rank=r.rank,
                    scores=r.retrieval_scores,
                )
                for r in results
            ],
        )
    except ValueError as exc:
        REQUESTS.labels(mode, "error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/feedback", status_code=201, dependencies=[Depends(authorize)])
async def feedback(request: Request, payload: FeedbackRequest):
    try:
        feedback_id = await asyncio.to_thread(
            request.app.state.search.feedback,
            payload.query_id,
            payload.document_id,
            payload.relevance,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown query_id") from exc
    return {"feedback_id": feedback_id, "status": "accepted"}

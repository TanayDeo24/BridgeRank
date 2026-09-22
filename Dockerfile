FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
ENV SEARCH_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 HF_HOME=/app/model-cache
RUN pip install --no-cache-dir ".[ml]"
RUN support-search build-index --output /app/artifacts/index/current
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 PORT=8000 SEARCH_ENVIRONMENT=production SEARCH_ARTIFACT_DIR=/app/artifacts/index/current SEARCH_DATABASE_PATH=/app/runtime/search.db SEARCH_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["sh", "-c", "uvicorn support_search.api:app --host 0.0.0.0 --port ${PORT}"]

from fastapi.testclient import TestClient

from support_search.api import app
from support_search.config import get_settings


def test_health_and_search(monkeypatch, tmp_path):
    database = tmp_path / "search.db"
    monkeypatch.setenv("SEARCH_DATABASE_PATH", str(database))
    monkeypatch.setenv("SEARCH_ARTIFACT_DIR", str(tmp_path / "missing"))
    get_settings.cache_clear()
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        response = client.get("/v1/search", params={"q": "wifi connection", "top_k": 2})
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2
        feedback = client.post(
            "/v1/feedback",
            json={
                "query_id": response.json()["query_id"],
                "document_id": response.json()["results"][0]["id"],
                "relevance": 3,
            },
        )
        assert feedback.status_code == 201
    assert database.exists()
    get_settings.cache_clear()

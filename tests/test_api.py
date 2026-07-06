from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app


def test_health():
    with patch("api.main.load_model", return_value="fake-model"):
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_check_returns_groundedness_score():
    with patch("api.main.load_model", return_value="fake-model"):
        with patch("api.main.check_groundedness", return_value={"grounded": True, "score": 0.91}):
            with TestClient(app) as client:
                response = client.post("/check", json={"context": "ctx", "answer": "ans"})
    assert response.status_code == 200
    assert response.json() == {"grounded": True, "score": 0.91}

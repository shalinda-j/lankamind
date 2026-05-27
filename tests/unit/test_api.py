"""
tests/unit/test_api.py
-----------------------
Unit tests for api.server using FastAPI's TestClient.

The /v1/complete endpoint calls run_client() which actually runs inference;
that is mocked out here so the tests run instantly without workers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.server import app
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data


# ── /v1/status ────────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_returns_200(self, client):
        resp = client.get("/v1/status")
        assert resp.status_code == 200

    def test_status_has_required_fields(self, client):
        resp = client.get("/v1/status")
        data = resp.json()
        for field in ("status", "uptime_seconds", "active_workers",
                      "healthy_workers", "requests_served"):
            assert field in data, f"Missing field: {field}"

    def test_status_ok_when_no_gateway(self, client):
        # With no gateway configured, should still return ok
        resp = client.get("/v1/status")
        assert resp.json()["status"] == "ok"
        assert resp.json()["active_workers"] == 0


# ── /v1/nodes ─────────────────────────────────────────────────────────────────

class TestNodes:
    def test_nodes_returns_200_no_gateway(self, client):
        resp = client.get("/v1/nodes")
        assert resp.status_code == 200

    def test_nodes_empty_list_no_gateway(self, client):
        resp = client.get("/v1/nodes")
        assert resp.json() == []


# ── /v1/complete ──────────────────────────────────────────────────────────────

class TestComplete:
    def test_complete_calls_run_client(self, client):
        with patch("cli.client.run_client", return_value="is beautiful.") as mock_rc:
            resp = client.post("/v1/complete", json={
                "prompt": "Sri Lanka",
                "max_tokens": 10,
            })
        assert resp.status_code == 200
        mock_rc.assert_called_once()

    def test_complete_response_fields(self, client):
        with patch("cli.client.run_client", return_value="is beautiful."):
            resp = client.post("/v1/complete", json={"prompt": "Sri Lanka"})
        data = resp.json()
        for field in ("prompt", "generated_text", "model", "tokens_generated", "elapsed_seconds"):
            assert field in data

    def test_complete_returns_generated_text(self, client):
        with patch("cli.client.run_client", return_value="is a beautiful island"):
            resp = client.post("/v1/complete", json={"prompt": "Sri Lanka"})
        assert resp.json()["generated_text"] == "is a beautiful island"

    def test_complete_echoes_prompt(self, client):
        with patch("cli.client.run_client", return_value="..."):
            resp = client.post("/v1/complete", json={"prompt": "Test prompt"})
        assert resp.json()["prompt"] == "Test prompt"

    def test_complete_503_when_inference_fails(self, client):
        with patch("cli.client.run_client", side_effect=RuntimeError("no workers")):
            resp = client.post("/v1/complete", json={"prompt": "Sri Lanka"})
        assert resp.status_code == 503

    def test_complete_missing_prompt_returns_422(self, client):
        resp = client.post("/v1/complete", json={"max_tokens": 10})
        assert resp.status_code == 422

    def test_complete_max_tokens_clamped(self, client):
        """max_tokens > 500 should be rejected (validation)."""
        with patch("cli.client.run_client", return_value="ok"):
            resp = client.post("/v1/complete", json={"prompt": "test", "max_tokens": 9999})
        assert resp.status_code == 422

    def test_complete_empty_prompt_rejected(self, client):
        with patch("cli.client.run_client", return_value="ok"):
            resp = client.post("/v1/complete", json={"prompt": ""})
        assert resp.status_code == 422

    def test_openapi_schema_reachable(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/v1/complete" in schema["paths"]

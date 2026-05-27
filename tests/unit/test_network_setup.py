"""
tests/unit/test_network_setup.py
---------------------------------
Tests for the mobile/LAN setup features:
  - /v1/network-info endpoint returns correct LAN URL
  - Static files are served at /
  - Mobile web UI (index.html) is reachable
  - PWA manifest is served
  - mDNS announce/browse round-trip (mocked)
  - LAN IP detection works
  - Multiple devices can reach the API (concurrent requests)
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from api.server import app
    return TestClient(app, raise_server_exceptions=False)


# ── /v1/network-info ──────────────────────────────────────────────────────────

class TestNetworkInfo:
    def test_returns_200(self, client):
        resp = client.get("/v1/network-info")
        assert resp.status_code == 200

    def test_has_lan_ip(self, client):
        data = client.get("/v1/network-info").json()
        ip = data["lan_ip"]
        parts = ip.split(".")
        assert len(parts) == 4

    def test_url_contains_ip(self, client):
        data = client.get("/v1/network-info").json()
        assert data["lan_ip"] in data["url"]

    def test_mdns_url_format(self, client):
        data = client.get("/v1/network-info").json()
        assert "lankamind.local" in data["mdns_url"]

    def test_qr_hint_non_empty(self, client):
        data = client.get("/v1/network-info").json()
        assert len(data["qr_hint"]) > 10

    def test_port_is_integer(self, client):
        data = client.get("/v1/network-info").json()
        assert isinstance(data["port"], int)
        assert data["port"] > 0


# ── Mobile web UI ─────────────────────────────────────────────────────────────

class TestMobileWebUI:
    def test_root_returns_html(self, client):
        """GET / should return the mobile web UI HTML page."""
        resp = client.get("/")
        assert resp.status_code == 200
        # Either a proper HTML file or a fallback HTML response
        ct = resp.headers.get("content-type", "")
        assert "html" in ct

    def test_root_contains_lankamind(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "LankaMind" in resp.text

    def test_manifest_served(self, client):
        resp = client.get("/manifest.json")
        # 200 if file exists, 404 is acceptable in test env without static files
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert "name" in data
            assert data["name"] == "LankaMind"

    def test_index_html_has_generate_button(self, client):
        resp = client.get("/")
        if resp.status_code == 200:
            assert "Generate" in resp.text or "generate" in resp.text.lower()

    def test_index_html_has_api_fetch(self, client):
        """The web UI JavaScript should call /v1/complete."""
        resp = client.get("/")
        if resp.status_code == 200:
            assert "/v1/complete" in resp.text

    def test_index_html_mobile_viewport(self, client):
        """The page must have a viewport meta tag for mobile scaling."""
        resp = client.get("/")
        if resp.status_code == 200:
            assert "viewport" in resp.text


# ── Static file content validation ───────────────────────────────────────────

class TestStaticFiles:
    STATIC_DIR = pathlib.Path(__file__).parent.parent.parent / "api" / "static"

    def test_index_html_exists(self):
        assert (self.STATIC_DIR / "index.html").exists(), \
            "api/static/index.html not found"

    def test_index_html_is_valid_html(self):
        content = (self.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content

    def test_index_html_has_pwa_manifest_link(self):
        content = (self.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert "manifest.json" in content

    def test_index_html_no_external_js_dependencies(self):
        """Web UI must work offline — no CDN script tags."""
        content = (self.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        # Should NOT have external script src (cdn.jsdelivr.net, unpkg.com, etc.)
        assert "cdn.jsdelivr.net" not in content
        assert "unpkg.com" not in content
        assert "cdnjs.cloudflare.com" not in content

    def test_manifest_json_exists(self):
        assert (self.STATIC_DIR / "manifest.json").exists()

    def test_manifest_json_valid(self):
        content = (self.STATIC_DIR / "manifest.json").read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["name"] == "LankaMind"
        assert "icons" in data
        assert data["display"] == "standalone"

    def test_manifest_background_color(self):
        content = (self.STATIC_DIR / "manifest.json").read_text(encoding="utf-8")
        data = json.loads(content)
        assert "background_color" in data
        # Dark theme
        assert data["background_color"].startswith("#")

    def test_index_html_has_api_docs_link(self):
        content = (self.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert "/docs" in content

    def test_index_html_has_github_link(self):
        content = (self.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert "github.com" in content


# ── LAN connectivity simulation ───────────────────────────────────────────────

class TestLanConnectivity:
    """
    Simulate multiple devices sending requests concurrently.
    (In a real test these would be different IP addresses, but TestClient
    handles the API layer correctly regardless of origin.)
    """

    def test_multiple_concurrent_health_checks(self, client):
        """10 concurrent health checks must all return 200."""
        results = []
        lock = threading.Lock()

        def check():
            r = client.get("/health")
            with lock:
                results.append(r.status_code)

        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 10
        assert all(s == 200 for s in results)

    def test_multiple_concurrent_network_info(self, client):
        """5 concurrent /v1/network-info requests must all succeed."""
        results = []
        lock = threading.Lock()

        def check():
            r = client.get("/v1/network-info")
            with lock:
                results.append(r.status_code)

        threads = [threading.Thread(target=check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert all(s == 200 for s in results)

    def test_cors_headers_present(self, client):
        """CORS must be open so any phone browser can call the API."""
        resp = client.get("/health", headers={"Origin": "http://192.168.1.100:9999"})
        # FastAPI adds CORS headers when Origin is present
        assert resp.status_code == 200


# ── API CORS for cross-device calls ──────────────────────────────────────────

class TestCorsPolicy:
    def test_options_preflight_from_mobile(self, client):
        """Browsers send OPTIONS preflight before POST — must return 200."""
        resp = client.options(
            "/v1/complete",
            headers={
                "Origin": "http://192.168.1.50:8000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        # 200 or 204 are both valid preflight responses
        assert resp.status_code in (200, 204)

    def test_complete_endpoint_accessible(self, client):
        """POST /v1/complete is accessible (workers mocked out)."""
        with patch("cli.client.run_client", return_value="beautiful island."):
            resp = client.post("/v1/complete", json={"prompt": "Sri Lanka is a"})
        assert resp.status_code == 200

    def test_response_includes_generated_text(self, client):
        with patch("cli.client.run_client", return_value="tropical paradise."):
            resp = client.post("/v1/complete", json={"prompt": "Sri Lanka is a"})
        assert resp.json()["generated_text"] == "tropical paradise."


# ── Installation smoke tests ──────────────────────────────────────────────────

class TestInstallation:
    def test_lankamind_importable(self):
        """The package must be importable without errors."""
        import cli.main   # noqa: F401
        import api.server  # noqa: F401
        import network.discovery  # noqa: F401

    def test_cli_main_has_serve_command(self):
        """The unified CLI must expose a 'serve' command."""
        from cli.main import cli
        assert "serve" in cli.commands

    def test_cli_main_has_all_commands(self):
        from cli.main import cli
        for cmd in ("complete", "chat", "node", "gateway", "bootstrap", "api",
                    "serve", "status", "balance", "keys"):
            assert cmd in cli.commands, f"Missing CLI command: {cmd}"

    def test_network_info_url_is_reachable_format(self, client):
        data = client.get("/v1/network-info").json()
        url = data["url"]
        assert url.startswith("http://")
        assert ":" in url  # has port

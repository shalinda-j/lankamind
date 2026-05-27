"""
tests/test_auto_worker.py
--------------------------
Unit tests for core.auto_worker helpers.
(The main auto_join() function is integration-level — tested via the CLI.)
"""
import pytest
from core.auto_worker import _local_ip, DISCOVER_TIMEOUT, TOPOLOGY_TIMEOUT


def test_local_ip_returns_string():
    ip = _local_ip()
    assert isinstance(ip, str)
    assert len(ip) > 0


def test_local_ip_looks_like_ip():
    ip = _local_ip()
    parts = ip.split(".")
    assert len(parts) == 4
    for p in parts:
        assert p.isdigit()


def test_timeouts_are_reasonable():
    assert DISCOVER_TIMEOUT >= 5
    assert TOPOLOGY_TIMEOUT >= 60


def test_discover_coordinator_returns_none_when_no_mdns(monkeypatch):
    """When mDNS browse finds nothing, should return None (not raise)."""
    from core.auto_worker import _discover_coordinator
    monkeypatch.setattr(
        "network.discovery.browse_once",
        lambda timeout=10: [],
    )
    result = _discover_coordinator(timeout=0.1)
    assert result is None


def test_discover_coordinator_finds_coord_service(monkeypatch):
    """Returns coordinator URL when mDNS finds a coordinator service."""
    from core.auto_worker import _discover_coordinator

    fake_services = [
        {"role": "coordinator", "host": "192.168.1.50", "coord_port": "5800"},
    ]
    monkeypatch.setattr(
        "network.discovery.browse_once",
        lambda timeout=10: fake_services,
    )
    url = _discover_coordinator(timeout=0.1)
    assert url == "http://192.168.1.50:5800"


def test_discover_coordinator_ignores_non_coordinator(monkeypatch):
    """Services with role != coordinator are ignored."""
    from core.auto_worker import _discover_coordinator

    fake_services = [
        {"role": "worker", "host": "192.168.1.10", "port": 5500},
        {"role": "gateway", "host": "192.168.1.10", "port": 5700},
    ]
    monkeypatch.setattr(
        "network.discovery.browse_once",
        lambda timeout=10: fake_services,
    )
    result = _discover_coordinator(timeout=0.1)
    assert result is None


def test_discover_coordinator_handles_mdns_exception(monkeypatch):
    """mDNS errors are caught and None is returned gracefully."""
    from core.auto_worker import _discover_coordinator

    def _raise(**kw):
        raise RuntimeError("mDNS unavailable")

    monkeypatch.setattr("network.discovery.browse_once", _raise)
    result = _discover_coordinator(timeout=0.1)
    assert result is None

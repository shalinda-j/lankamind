"""
tests/unit/test_registry.py
----------------------------
Unit tests for orchestrator.registry.NodeRegistry and WorkerInfo.
"""

from __future__ import annotations

import time

import pytest

from orchestrator.registry import NodeRegistry, WorkerInfo


# ── WorkerInfo helpers ────────────────────────────────────────────────────────


def make_worker(**kwargs) -> WorkerInfo:
    defaults = dict(
        worker_id="w1",
        shard_idx=0,
        num_shards=3,
        model_name="gpt2",
        host="localhost",
        port=5500,
    )
    defaults.update(kwargs)
    return WorkerInfo(**defaults)


# ── WorkerInfo tests ──────────────────────────────────────────────────────────


class TestWorkerInfo:
    def test_address(self):
        w = make_worker(host="192.168.1.5", port=5500)
        assert w.address == "tcp://192.168.1.5:5500"

    def test_ping_address(self):
        w = make_worker(port=5500)
        assert w.ping_address == "tcp://localhost:5600"

    def test_not_stale_when_fresh(self):
        w = make_worker()
        assert not w.is_stale(timeout_s=15.0)

    def test_stale_when_old(self):
        w = make_worker()
        w.last_heartbeat = time.time() - 20.0
        assert w.is_stale(timeout_s=15.0)

    def test_to_dict_has_required_keys(self):
        w = make_worker()
        d = w.to_dict()
        for key in ("worker_id", "shard_idx", "num_shards", "model_name",
                    "host", "port", "latency_ms", "is_healthy"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values(self):
        w = make_worker(worker_id="abc", shard_idx=1, port=5501, latency_ms=42.0)
        d = w.to_dict()
        assert d["worker_id"] == "abc"
        assert d["shard_idx"] == 1
        assert d["port"] == 5501
        assert d["latency_ms"] == 42.0


# ── NodeRegistry tests ────────────────────────────────────────────────────────


class TestNodeRegistry:
    def _make_registry(self) -> NodeRegistry:
        return NodeRegistry()

    def _register_worker(
        self,
        reg: NodeRegistry,
        worker_id: str = "w1",
        shard_idx: int = 0,
        num_shards: int = 3,
        model_name: str = "gpt2",
        host: str = "localhost",
        port: int = 5500,
    ) -> WorkerInfo:
        return reg.register(
            worker_id=worker_id,
            shard_idx=shard_idx,
            num_shards=num_shards,
            model_name=model_name,
            host=host,
            port=port,
        )

    def test_register_returns_worker_info(self):
        reg = self._make_registry()
        w = self._register_worker(reg)
        assert isinstance(w, WorkerInfo)
        assert w.worker_id == "w1"

    def test_register_idempotent(self):
        """Re-registering same worker_id should update, not duplicate."""
        reg = self._make_registry()
        self._register_worker(reg, worker_id="w1")
        self._register_worker(reg, worker_id="w1")
        workers = reg.get_workers()
        assert len(workers) == 1

    def test_get_workers_empty(self):
        reg = self._make_registry()
        assert reg.get_workers() == []

    def test_get_workers_returns_all(self):
        reg = self._make_registry()
        self._register_worker(reg, worker_id="w0", shard_idx=0, port=5500)
        self._register_worker(reg, worker_id="w1", shard_idx=1, port=5501)
        self._register_worker(reg, worker_id="w2", shard_idx=2, port=5502)
        workers = reg.get_workers()
        assert len(workers) == 3

    def test_get_workers_filter_by_model(self):
        reg = self._make_registry()
        self._register_worker(reg, worker_id="a", model_name="gpt2")
        self._register_worker(reg, worker_id="b", model_name="gpt2-medium")
        gpt2 = reg.get_workers(model_name="gpt2")
        assert len(gpt2) == 1
        assert gpt2[0].worker_id == "a"

    def test_get_workers_excludes_unhealthy_by_default(self):
        reg = self._make_registry()
        self._register_worker(reg, worker_id="w1")
        reg.mark_unhealthy("w1")
        assert reg.get_workers(healthy_only=True) == []

    def test_get_workers_includes_unhealthy_when_asked(self):
        reg = self._make_registry()
        self._register_worker(reg, worker_id="w1")
        reg.mark_unhealthy("w1")
        workers = reg.get_workers(healthy_only=False)
        assert len(workers) == 1

    def test_update_latency(self):
        reg = self._make_registry()
        self._register_worker(reg, worker_id="w1")
        reg.update_latency("w1", 12.5)
        w = reg.get_workers()[0]
        assert w.latency_ms == pytest.approx(12.5)

    def test_update_latency_unknown_worker_no_error(self):
        reg = self._make_registry()
        # Should not raise even if worker_id doesn't exist
        reg.update_latency("nonexistent", 5.0)

    def test_mark_unhealthy(self):
        reg = self._make_registry()
        self._register_worker(reg, worker_id="w1")
        reg.mark_unhealthy("w1")
        workers = reg.get_workers(healthy_only=False)
        assert not workers[0].is_healthy

    def test_stale_workers_evicted(self):
        reg = self._make_registry()
        w = self._register_worker(reg, worker_id="w1")
        # Manually age the heartbeat timestamp
        w.last_heartbeat = time.time() - 60.0
        # Trigger eviction via get_workers
        workers = reg.get_workers()
        assert workers == []

    def test_snapshot_returns_list_of_dicts(self):
        reg = self._make_registry()
        self._register_worker(reg, worker_id="w1")
        snap = reg.snapshot()
        assert isinstance(snap, list)
        assert isinstance(snap[0], dict)
        assert snap[0]["worker_id"] == "w1"

    def test_snapshot_empty(self):
        reg = self._make_registry()
        assert reg.snapshot() == []

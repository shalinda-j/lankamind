"""
tests/unit/test_scheduler.py
-----------------------------
Unit tests for orchestrator.scheduler.Scheduler.
"""

from __future__ import annotations

import pytest

from orchestrator.registry import NodeRegistry, WorkerInfo
from orchestrator.scheduler import NotEnoughWorkersError, Scheduler


# ── Helpers ───────────────────────────────────────────────────────────────────


def populate_registry(
    reg: NodeRegistry,
    num_shards: int = 3,
    model_name: str = "gpt2",
    base_port: int = 5500,
) -> None:
    """Register a complete, healthy set of workers."""
    for i in range(num_shards):
        reg.register(
            worker_id=f"w{i}",
            shard_idx=i,
            num_shards=num_shards,
            model_name=model_name,
            host="localhost",
            port=base_port + i,
        )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestScheduler:
    def _make(self) -> tuple[NodeRegistry, Scheduler]:
        reg = NodeRegistry()
        sched = Scheduler(reg)
        return reg, sched

    # -- Happy path -----------------------------------------------------------

    def test_build_pipeline_returns_correct_count(self):
        reg, sched = self._make()
        populate_registry(reg, num_shards=3)
        chain = sched.build_pipeline("gpt2", 3)
        assert len(chain) == 3

    def test_build_pipeline_correct_shard_order(self):
        reg, sched = self._make()
        populate_registry(reg, num_shards=3)
        chain = sched.build_pipeline("gpt2", 3)
        for i, w in enumerate(chain):
            assert w.shard_idx == i, f"Position {i} has shard_idx {w.shard_idx}"

    def test_build_pipeline_returns_worker_infos(self):
        reg, sched = self._make()
        populate_registry(reg, num_shards=3)
        chain = sched.build_pipeline("gpt2", 3)
        for w in chain:
            assert isinstance(w, WorkerInfo)

    def test_build_pipeline_2_shards(self):
        reg, sched = self._make()
        populate_registry(reg, num_shards=2)
        chain = sched.build_pipeline("gpt2", 2)
        assert len(chain) == 2
        assert chain[0].shard_idx == 0
        assert chain[1].shard_idx == 1

    def test_build_pipeline_picks_lowest_latency(self):
        """When two workers serve shard 0, the lower latency one should win."""
        reg, sched = self._make()
        populate_registry(reg, num_shards=3)

        # Add a second worker for shard 0 with lower latency
        reg.register(
            worker_id="w0_fast",
            shard_idx=0,
            num_shards=3,
            model_name="gpt2",
            host="localhost",
            port=5510,
        )
        reg.update_latency("w0", 999.0)
        reg.update_latency("w0_fast", 1.0)

        chain = sched.build_pipeline("gpt2", 3)
        assert chain[0].worker_id == "w0_fast"

    def test_build_pipeline_multiple_models_isolated(self):
        """Workers for 'gpt2-medium' should not be used for 'gpt2' pipeline."""
        reg, sched = self._make()
        populate_registry(reg, num_shards=3, model_name="gpt2")
        populate_registry(reg, num_shards=3, model_name="gpt2-medium", base_port=5600)
        chain = sched.build_pipeline("gpt2", 3)
        for w in chain:
            assert w.model_name == "gpt2"

    # -- Error cases ----------------------------------------------------------

    def test_raises_when_no_workers(self):
        reg, sched = self._make()
        with pytest.raises(NotEnoughWorkersError):
            sched.build_pipeline("gpt2", 3)

    def test_raises_when_missing_shard(self):
        reg, sched = self._make()
        # Register shards 0 and 1 but not shard 2
        reg.register(worker_id="w0", shard_idx=0, num_shards=3, model_name="gpt2",
                     host="localhost", port=5500)
        reg.register(worker_id="w1", shard_idx=1, num_shards=3, model_name="gpt2",
                     host="localhost", port=5501)
        with pytest.raises(NotEnoughWorkersError):
            sched.build_pipeline("gpt2", 3)

    def test_raises_when_all_unhealthy(self):
        reg, sched = self._make()
        populate_registry(reg, num_shards=3)
        for i in range(3):
            reg.mark_unhealthy(f"w{i}")
        with pytest.raises(NotEnoughWorkersError):
            sched.build_pipeline("gpt2", 3)

    def test_raises_wrong_model(self):
        reg, sched = self._make()
        populate_registry(reg, model_name="gpt2-medium", num_shards=3)
        with pytest.raises(NotEnoughWorkersError):
            sched.build_pipeline("gpt2", 3)

    def test_error_message_is_informative(self):
        reg, sched = self._make()
        with pytest.raises(NotEnoughWorkersError, match="gpt2"):
            sched.build_pipeline("gpt2", 3)

    def test_raises_when_num_shards_mismatch(self):
        """Workers registered for 3-shard config should not serve 4-shard request."""
        reg, sched = self._make()
        populate_registry(reg, num_shards=3)
        with pytest.raises(NotEnoughWorkersError):
            sched.build_pipeline("gpt2", 4)

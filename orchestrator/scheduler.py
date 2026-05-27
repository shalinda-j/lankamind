"""
orchestrator/scheduler.py
--------------------------
Picks the best chain of workers for an inference request.

Algorithm (Phase 2 — simple greedy):
  1. Fetch all healthy workers for the requested model.
  2. Keep only workers that match num_shards (so a 3-shard request
     won't accidentally use a worker configured for 4 shards).
  3. For each shard slot 0 … N-1, pick the worker with the lowest
     measured latency (updated by the HealthChecker).
  4. Return the ordered list [shard0_worker, shard1_worker, …].

Raises NotEnoughWorkersError if any slot is uncovered.

Phase 3 upgrade: factor in inter-node latency (not just node latency)
so we prefer chains where consecutive nodes are geographically close.
"""

from __future__ import annotations

from typing import Dict, List

from orchestrator.registry import NodeRegistry, WorkerInfo


class NotEnoughWorkersError(RuntimeError):
    """Raised when the registry can't fill all shard slots."""
    pass


class Scheduler:
    def __init__(self, registry: NodeRegistry):
        self.registry = registry

    def build_pipeline(self, model_name: str, num_shards: int) -> List[WorkerInfo]:
        """
        Return an ordered list of workers [shard_0, shard_1, ..., shard_N-1].

        Each element is the lowest-latency healthy worker for that slot.
        Raises NotEnoughWorkersError if any slot has no healthy worker.
        """
        workers = self.registry.get_workers(model_name, healthy_only=True)

        # Only consider workers that agree on the pipeline width
        compatible = [w for w in workers if w.num_shards == num_shards]

        # For each slot keep the worker with the lowest measured latency
        slots: Dict[int, WorkerInfo] = {}
        for w in compatible:
            if w.shard_idx not in slots or w.latency_ms < slots[w.shard_idx].latency_ms:
                slots[w.shard_idx] = w

        missing = [i for i in range(num_shards) if i not in slots]
        if missing:
            raise NotEnoughWorkersError(
                f"Cannot build a {num_shards}-shard pipeline for '{model_name}': "
                f"missing shards {missing}.  "
                f"Start more workers with:\n"
                f"  python scripts/launch_workers.py "
                f"--shards {num_shards} --model {model_name}"
            )

        return [slots[i] for i in range(num_shards)]

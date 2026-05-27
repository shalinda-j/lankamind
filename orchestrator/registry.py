"""
orchestrator/registry.py
-------------------------
Node registry — tracks which devices are online and what shards they hold.
(Phase 2 — stub only.)

Phase 2 plan:
  • Each worker pings a central registry on startup with:
      { peer_id, shard_idx, model_name, latency_ms, ram_free_gb }
  • Registry stores entries with a TTL; entries expire if the worker stops
    sending heartbeats.
  • Scheduler (scheduler.py) queries this registry to build a pipeline.
"""


class NodeRegistry:
    def register(self, peer_id: str, metadata: dict) -> None:
        raise NotImplementedError("Node registry is planned for Phase 2.")

    def get_available_shards(self, model_name: str) -> list:
        raise NotImplementedError("Node registry is planned for Phase 2.")

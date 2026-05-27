"""
orchestrator/scheduler.py
--------------------------
Query scheduler — picks the fastest available chain of nodes for a request.
(Phase 2 — stub only.)

Phase 2 plan:
  • Reads the live node list from NodeRegistry.
  • Builds a pipeline (ordered list of peer IDs, one per shard) that minimises
    estimated latency (based on measured round-trip time between nodes).
  • Falls back to the next-best chain if any node in the chosen chain goes
    offline mid-request (Phase 2 stretch goal).
"""


class Scheduler:
    def build_pipeline(self, model_name: str, num_shards: int) -> list[str]:
        raise NotImplementedError("Scheduler is planned for Phase 2.")

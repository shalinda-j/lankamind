"""
orchestrator/load_balancer.py
------------------------------
Load balancer — distributes incoming requests across multiple pipelines.
(Phase 2 — stub only.)

Phase 2 plan:
  • Maintains a pool of active pipelines.
  • Routes each incoming request to the least-loaded pipeline.
  • Monitors pipeline health and removes broken pipelines from the pool.
"""


class LoadBalancer:
    def route(self, request_id: str) -> str:
        raise NotImplementedError("Load balancer is planned for Phase 2.")

"""
trust/reputation.py
--------------------
Per-node reputation score.  (Phase 4 — stub only.)

Phase 4 plan:
  • Each completed inference request is checked: did the node return the
    correct output? (verified by spot-checking against a known reference).
  • Reputation score = weighted moving average of success rate.
  • Nodes with low reputation are demoted in the scheduler's ranking.
  • Score is stored on-chain (or in a signed off-chain ledger) so it
    cannot be self-reported dishonestly.
"""


class ReputationTracker:
    def record_success(self, peer_id: str) -> None:
        raise NotImplementedError("Reputation tracking is planned for Phase 4.")

    def record_failure(self, peer_id: str) -> None:
        raise NotImplementedError("Reputation tracking is planned for Phase 4.")

    def get_score(self, peer_id: str) -> float:
        raise NotImplementedError("Reputation tracking is planned for Phase 4.")

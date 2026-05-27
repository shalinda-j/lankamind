"""
trust/ledger.py
----------------
Token reward ledger — tracks earnings for nodes that contribute compute.
(Phase 4 — stub only.)

Phase 4 plan:
  • v1: Off-chain signed ledger (simple JSON file on a coordinator server).
        Pros: no gas fees, instant settlement, easy to prototype.
        Cons: requires trusting the coordinator.

  • v2: On-chain smart contract on an EVM-compatible L2 (e.g. Polygon).
        Pros: trustless, auditable, enables open participation.
        Cons: gas fees, requires wallet infrastructure.

  • Reward formula (draft):
        reward = tokens_processed × latency_bonus × reputation_multiplier
"""


class RewardLedger:
    def credit(self, peer_id: str, amount: float) -> None:
        raise NotImplementedError("Reward ledger is planned for Phase 4.")

    def balance(self, peer_id: str) -> float:
        raise NotImplementedError("Reward ledger is planned for Phase 4.")

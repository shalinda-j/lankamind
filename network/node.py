"""
network/node.py
---------------
Full P2P node abstraction.  (Phase 3 — stub only.)

Phase 3 plan:
  • Each device runs a Node that holds a libp2p Host.
  • The node advertises its capabilities (model, shard range, RAM, GPU) to
    the DHT.
  • Accepts incoming inference streams from upstream nodes.
  • Maintains persistent connections to downstream nodes.
"""


class Node:
    def start(self) -> None:
        raise NotImplementedError("P2P Node is planned for Phase 3.")

    def stop(self) -> None:
        raise NotImplementedError("P2P Node is planned for Phase 3.")

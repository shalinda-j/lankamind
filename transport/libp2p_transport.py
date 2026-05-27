"""
transport/libp2p_transport.py
------------------------------
libp2p-based transport for real P2P networking.  (Phase 3 — stub only.)

WHY LIBP2P?
  ZeroMQ (Phase 1) is great for local processes and LAN deployments.
  libp2p is the same protocol family that powers IPFS, Filecoin, and Petals.
  It gives us:
    • Peer discovery via DHT (devices find each other without a central server)
    • NAT traversal (works behind home routers)
    • Multiplexed streams (multiple logical channels over one connection)
    • Connection encryption (noise protocol)

Phase 3 plan:
  1. pip install py-libp2p (or use hivemind which wraps it)
  2. Implement Libp2pInputSocket and Libp2pOutputSocket with the same
     recv() / send() interface as ZMQ sockets so the rest of the code
     doesn't change.
  3. Add DHT-based peer discovery in network/discovery.py.
  4. Update PipelineConfig to hold peer IDs instead of port numbers.
"""

# TODO (Phase 3): implement Libp2pInputSocket and Libp2pOutputSocket


class Libp2pInputSocket:
    def recv(self):
        raise NotImplementedError("libp2p transport is planned for Phase 3.")


class Libp2pOutputSocket:
    def send(self, header, tensor=None):
        raise NotImplementedError("libp2p transport is planned for Phase 3.")

"""
network/discovery.py
---------------------
Peer discovery via a distributed hash table (DHT).  (Phase 3 — stub only.)

Phase 3 plan:
  • Use hivemind's DHT (built on libp2p Kademlia) to let nodes find each
    other without a central server.
  • Any new node announces itself with:  dht.store(model_key, peer_addr, ttl=60s)
  • Clients query:  dht.get(model_key) → list of peer addresses
  • Works across home routers via NAT hole-punching.
"""


class PeerDiscovery:
    def announce(self, model_name: str, peer_addr: str) -> None:
        raise NotImplementedError("Peer discovery is planned for Phase 3.")

    def find_peers(self, model_name: str) -> list[str]:
        raise NotImplementedError("Peer discovery is planned for Phase 3.")

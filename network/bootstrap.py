"""
network/bootstrap.py
--------------------
Lightweight bootstrap / peer-discovery node for LankaMind.

The bootstrap node acts as a well-known rendezvous point so that new worker
nodes can find each other without a centralised registry.  It is intentionally
minimal — it only stores {peer_id → (host, port, last_seen)} and returns the
live peer list on request.

Protocol (ZMQ REP on bootstrap_port)
--------------------------------------
Register:
    REQ: {"type": "register", "peer_id": str, "host": str, "port": int}
    REP: {"status": "ok"}

List peers:
    REQ: {"type": "list"}
    REP: {"status": "ok", "peers": [{"peer_id": str, "host": str, "port": int}, ...]}

Ping:
    REQ: {"type": "ping"}
    REP: {"status": "ok", "pong": true}

Design notes
------------
• Peers expire after PEER_TTL_SECONDS (60 s) of inactivity — they must re-register
  periodically (every ~30 s) to stay in the list.
• A separate heartbeat thread evicts stale peers every EVICT_INTERVAL_SECONDS (15 s).
• The node is intentionally NOT the gateway — the gateway handles model-specific
  chain assembly.  The bootstrap node just helps new peers discover the gateway's
  address.

Run:
    python -m network.bootstrap --port 6000

Or via scripts/launch_bootstrap.py.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

import zmq

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[Bootstrap] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_PORT = 6000
PEER_TTL_SECONDS = 60.0
EVICT_INTERVAL_SECONDS = 15.0


@dataclass
class PeerInfo:
    peer_id: str
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)

    def is_stale(self, ttl: float = PEER_TTL_SECONDS) -> bool:
        return (time.time() - self.last_seen) > ttl

    def to_dict(self) -> dict:
        return {"peer_id": self.peer_id, "host": self.host, "port": self.port}


class PeerRegistry:
    """Thread-safe in-memory registry of bootstrap peers."""

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._peers: Dict[str, PeerInfo] = {}

    def register(self, peer_id: str, host: str, port: int) -> None:
        with self._lock:
            if peer_id in self._peers:
                p = self._peers[peer_id]
                p.host = host
                p.port = port
                p.last_seen = time.time()
            else:
                self._peers[peer_id] = PeerInfo(peer_id=peer_id, host=host, port=port)

    def list_peers(self) -> List[dict]:
        with self._lock:
            self._evict()
            return [p.to_dict() for p in self._peers.values()]

    def evict_stale(self) -> int:
        with self._lock:
            return self._evict()

    def _evict(self) -> int:
        stale = [pid for pid, p in self._peers.items() if p.is_stale()]
        for pid in stale:
            del self._peers[pid]
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._peers)


class BootstrapNode:
    """
    Runs a ZMQ REP server that handles peer registration and listing.
    """

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self.port = port
        self.registry = PeerRegistry()
        self._stop = threading.Event()

    # ── Eviction background thread ────────────────────────────────────────────

    def _evict_loop(self) -> None:
        while not self._stop.wait(EVICT_INTERVAL_SECONDS):
            evicted = self.registry.evict_stale()
            if evicted:
                log.info("Evicted %d stale peer(s). Active: %d", evicted, len(self.registry))

    # ── Request handler ───────────────────────────────────────────────────────

    def _handle(self, req: dict) -> dict:
        req_type = req.get("type", "")

        if req_type == "register":
            self.registry.register(
                peer_id=req["peer_id"],
                host=req["host"],
                port=req["port"],
            )
            log.info("Registered peer %s@%s:%d", req["peer_id"], req["host"], req["port"])
            return {"status": "ok"}

        elif req_type == "list":
            peers = self.registry.list_peers()
            return {"status": "ok", "peers": peers}

        elif req_type == "ping":
            return {"status": "ok", "pong": True}

        else:
            return {"status": "error", "message": f"Unknown type: {req_type!r}"}

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.RCVTIMEO = 1_000
        sock.bind(f"tcp://*:{self.port}")
        log.info("Bootstrap node listening on port %d", self.port)

        evict_thread = threading.Thread(
            target=self._evict_loop, daemon=True, name="Evict"
        )
        evict_thread.start()

        def _shutdown(*_: object) -> None:
            log.info("Shutting down bootstrap node…")
            self._stop.set()
            sock.close()
            ctx.term()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown)

        try:
            while not self._stop.is_set():
                try:
                    req = sock.recv_json()
                except zmq.Again:
                    continue
                except Exception as exc:
                    log.warning("recv error: %s", exc)
                    continue

                try:
                    resp = self._handle(req)
                except Exception as exc:
                    resp = {"status": "error", "message": str(exc)}

                try:
                    sock.send_json(resp)
                except Exception as exc:
                    log.warning("send error: %s", exc)

        except KeyboardInterrupt:
            log.info("Keyboard interrupt.")
        finally:
            self._stop.set()
            sock.close()
            ctx.term()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="LankaMind bootstrap / peer-discovery node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="ZMQ REP port to listen on")
    args = p.parse_args()
    BootstrapNode(port=args.port).run()


if __name__ == "__main__":
    main()

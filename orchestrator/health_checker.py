"""
orchestrator/health_checker.py
-------------------------------
Background thread that actively pings each registered worker.

Every CHECK_INTERVAL_S seconds it sends a "ping" to each worker's
ping port (inference_port + 100).  Workers that reply "pong" within
PING_TIMEOUT_MS milliseconds have their latency updated.  Workers that
miss MISS_LIMIT consecutive pings are marked unhealthy.

Design choice: separate ping port (input_port + 100) rather than
sharing the inference PULL socket.  This keeps health-check traffic
completely isolated from the inference pipeline — a slow inference
doesn't delay health checks and a missed ping never corrupts activations.

Note: if a worker doesn't have a ping responder running (e.g., old Phase 1
workers without --gateway-address), pings will always time out and the
worker will eventually be marked unhealthy.  That is the correct behaviour.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import zmq

from orchestrator.registry import NodeRegistry

CHECK_INTERVAL_S = 10.0
PING_TIMEOUT_MS  = 3_000
MISS_LIMIT       = 3


class HealthChecker:
    def __init__(self, registry: NodeRegistry):
        self.registry = registry
        self._thread: Optional[threading.Thread] = None
        self._stop    = threading.Event()
        self._misses: dict[str, int] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="HealthChecker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.wait(CHECK_INTERVAL_S):
            self._check_all()

    def _check_all(self) -> None:
        snapshot = self.registry.snapshot()
        if not snapshot:
            return

        ctx = zmq.Context()
        try:
            for w in snapshot:
                worker_id  = w["worker_id"]
                ping_addr  = f"tcp://{w['host']}:{w['port'] + 100}"
                latency_ms = self._ping(ctx, ping_addr)

                if latency_ms is not None:
                    self.registry.update_latency(worker_id, latency_ms)
                    self._misses[worker_id] = 0
                else:
                    self._misses[worker_id] = self._misses.get(worker_id, 0) + 1
                    if self._misses[worker_id] >= MISS_LIMIT:
                        self.registry.mark_unhealthy(worker_id)
        finally:
            ctx.term()

    def _ping(self, ctx: zmq.Context, address: str) -> Optional[float]:
        """
        Send "ping" to address, return round-trip latency in ms or None on timeout.
        """
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, PING_TIMEOUT_MS)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(address)
        try:
            t0 = time.monotonic()
            sock.send_string("ping")
            response = sock.recv_string()
            if response == "pong":
                return (time.monotonic() - t0) * 1000.0
        except zmq.Again:
            pass   # timeout
        except Exception:
            pass
        finally:
            sock.close()
        return None

"""
orchestrator/registry.py
-------------------------
Thread-safe node registry with heartbeat-based expiry.

Workers "check in" by sending a JSON heartbeat to the gateway every 5 s.
Any worker that hasn't been heard from in stale_timeout_s seconds is
silently evicted from the pool — the scheduler will never see it.

Design choice: in-memory only (no database).
  Pro:  zero latency reads, no external dep, simple.
  Con:  state is lost if the gateway restarts.
  Phase 3+ upgrade: persist to a local SQLite file so the gateway
  can reload known nodes after a restart.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Worker record ─────────────────────────────────────────────────────────────


@dataclass
class WorkerInfo:
    """All we know about one worker node."""

    worker_id: str
    shard_idx: int
    num_shards: int
    model_name: str
    host: str
    port: int
    last_heartbeat: float = field(default_factory=time.time)
    latency_ms: float = 9999.0   # updated by HealthChecker
    is_healthy: bool = True
    requests_processed: int = 0

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def address(self) -> str:
        """ZMQ address clients connect to."""
        return f"tcp://{self.host}:{self.port}"

    @property
    def ping_address(self) -> str:
        """Separate ping port = inference port + 100."""
        return f"tcp://{self.host}:{self.port + 100}"

    def is_stale(self, timeout_s: float = 15.0) -> bool:
        return (time.time() - self.last_heartbeat) > timeout_s

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (used in API responses)."""
        return {
            "worker_id":          self.worker_id,
            "shard_idx":          self.shard_idx,
            "num_shards":         self.num_shards,
            "model_name":         self.model_name,
            "host":               self.host,
            "port":               self.port,
            "address":            self.address,
            "last_heartbeat":     self.last_heartbeat,
            "latency_ms":         round(self.latency_ms, 1),
            "is_healthy":         self.is_healthy,
            "requests_processed": self.requests_processed,
        }


# ── Registry ──────────────────────────────────────────────────────────────────


class NodeRegistry:
    """
    Thread-safe, heartbeat-based worker registry.

    All public methods are safe to call from multiple threads simultaneously.
    Stale entries are evicted lazily (on the next read) rather than by a
    background thread, to keep the implementation simple.
    """

    def __init__(self, stale_timeout_s: float = 15.0):
        self._workers: Dict[str, WorkerInfo] = {}
        self._lock = threading.Lock()
        self.stale_timeout_s = stale_timeout_s

    # ── Writes ────────────────────────────────────────────────────────────────

    def register(
        self,
        worker_id: str,
        shard_idx: int,
        num_shards: int,
        model_name: str,
        host: str,
        port: int,
        requests_processed: int = 0,
    ) -> WorkerInfo:
        """
        Register a new worker or refresh the heartbeat of an existing one.
        Called each time a heartbeat arrives.
        """
        with self._lock:
            if worker_id in self._workers:
                w = self._workers[worker_id]
                w.last_heartbeat = time.time()
                w.is_healthy = True
                w.requests_processed = requests_processed
            else:
                self._workers[worker_id] = WorkerInfo(
                    worker_id=worker_id,
                    shard_idx=shard_idx,
                    num_shards=num_shards,
                    model_name=model_name,
                    host=host,
                    port=port,
                    requests_processed=requests_processed,
                )
            return self._workers[worker_id]

    def update_latency(self, worker_id: str, latency_ms: float) -> None:
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].latency_ms = latency_ms

    def mark_unhealthy(self, worker_id: str) -> None:
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].is_healthy = False

    def remove(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_workers(
        self,
        model_name: Optional[str] = None,
        healthy_only: bool = True,
    ) -> List[WorkerInfo]:
        """
        Return all workers, optionally filtered.
        Stale entries are evicted before returning.
        """
        self._evict_stale()
        with self._lock:
            workers = list(self._workers.values())

        if model_name and model_name != "*":
            workers = [w for w in workers if w.model_name == model_name]
        if healthy_only:
            workers = [w for w in workers if w.is_healthy]

        return sorted(workers, key=lambda w: w.shard_idx)

    def snapshot(self) -> List[dict]:
        """JSON-serialisable snapshot of the full registry."""
        self._evict_stale()
        with self._lock:
            return [w.to_dict() for w in self._workers.values()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._workers)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict_stale(self) -> None:
        """Remove workers whose last heartbeat is too old."""
        with self._lock:
            stale = [
                wid for wid, w in self._workers.items()
                if w.is_stale(self.stale_timeout_s)
            ]
            for wid in stale:
                del self._workers[wid]

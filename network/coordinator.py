"""
network/coordinator.py
----------------------
LankaMind Network Coordinator

Manages shard assignments so every device that runs
``lankamind join`` is automatically configured and wired into the pipeline.

Architecture
============
One coordinator per LankaMind network.  The first machine starts it with
``lankamind start``.  Every other device runs ``lankamind join``.

Flow for a joining device
-------------------------
1. Device discovers coordinator via mDNS (``_lankamind-coord._tcp.local.``)
   or is given the URL directly.
2. Device POSTs to ``/api/register`` with its LAN IP.
3. Coordinator assigns a shard index and records the worker's IP.
4. When all N shards have registered, coordinator builds the routing table
   (each shard's output_address = next shard's IP:port).
5. Workers GET ``/api/config/{shard_idx}`` — 425 while topology is incomplete,
   200 once ready.
6. Each worker starts with the returned config and signals ready via
   ``POST /api/ready/{shard_idx}``.
7. Coordinator marks ``complete=true`` once every shard is ready.

REST API
--------
POST /api/register   {host}          → {shard_idx, num_shards, input_port, model, ...}
GET  /api/topology                   → {workers, registered, complete, slots_remaining}
GET  /api/config/{shard_idx}         → {input_port, output_address, model}  (425 while incomplete)
POST /api/ready/{shard_idx}          → {ok, shard_idx}
GET  /health                         → {status, registered, ready, complete}
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

BASE_WORKER_PORT  = 5500
RESULT_PORT       = 5599
DEFAULT_COORD_PORT = 5800


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class WorkerSlot:
    shard_idx:     int
    host:          str
    input_port:    int
    registered_at: float = field(default_factory=time.time)
    ready:         bool  = False


# ── Core coordinator logic ────────────────────────────────────────────────────


class NetworkCoordinator:
    """
    Thread-safe coordinator that assigns shards and builds routing tables.
    Independent of any HTTP framework — tested directly.
    """

    def __init__(
        self,
        num_shards:  int,
        model:       str,
        result_host: str = "localhost",
    ) -> None:
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        self.num_shards  = num_shards
        self.model       = model
        self.result_host = result_host
        self._lock       = threading.Lock()
        self._slots:     list[WorkerSlot] = []
        self._next_shard = 0

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, host: str) -> dict:
        """
        Assign the next available shard to a worker at *host*.
        Returns a dict with shard_idx, num_shards, input_port, model.
        Returns ``{"error": "network_full"}`` if all slots are taken.
        """
        with self._lock:
            if self._next_shard >= self.num_shards:
                return {"error": "network_full", "shard_idx": -1}

            shard_idx  = self._next_shard
            self._next_shard += 1
            input_port = BASE_WORKER_PORT + shard_idx

            self._slots.append(WorkerSlot(
                shard_idx=shard_idx, host=host, input_port=input_port,
            ))

            log.info(
                "Registered shard %d/%d from %s (input port %d)",
                shard_idx, self.num_shards - 1, host, input_port,
            )

            return {
                "shard_idx":       shard_idx,
                "num_shards":      self.num_shards,
                "model":           self.model,
                "input_port":      input_port,
                "status":          "assigned",
                "slots_remaining": self.num_shards - self._next_shard,
            }

    # ── Topology / routing ────────────────────────────────────────────────────

    def get_config(self, shard_idx: int) -> dict | None:
        """
        Return the full routing config for a shard once ALL workers have
        registered (so we know every IP:port).
        Returns None while topology is incomplete.
        """
        with self._lock:
            if len(self._slots) < self.num_shards:
                return None  # not all registered yet

            slot = next((s for s in self._slots if s.shard_idx == shard_idx), None)
            if slot is None:
                return None

            # Build output_address: point to the *next* shard's host:port
            if shard_idx < self.num_shards - 1:
                next_slot = next(
                    (s for s in self._slots if s.shard_idx == shard_idx + 1), None
                )
                if next_slot:
                    output_address = f"tcp://{next_slot.host}:{next_slot.input_port}"
                else:
                    # Fallback — shouldn't happen
                    output_address = f"tcp://localhost:{BASE_WORKER_PORT + shard_idx + 1}"
            else:
                # Last shard → result collector on the coordinator's machine
                output_address = f"tcp://{self.result_host}:{RESULT_PORT}"

            return {
                "shard_idx":      shard_idx,
                "num_shards":     self.num_shards,
                "model":          self.model,
                "input_port":     slot.input_port,
                "output_address": output_address,
                "host":           slot.host,
            }

    def mark_ready(self, shard_idx: int) -> bool:
        """Mark a shard as ready (model loaded, inference loop running)."""
        with self._lock:
            for slot in self._slots:
                if slot.shard_idx == shard_idx:
                    slot.ready = True
                    log.info("Shard %d marked ready", shard_idx)
                    return True
        return False

    # ── Status helpers ────────────────────────────────────────────────────────

    @property
    def all_registered(self) -> bool:
        with self._lock:
            return len(self._slots) >= self.num_shards

    @property
    def all_ready(self) -> bool:
        with self._lock:
            return (
                len(self._slots) >= self.num_shards
                and all(s.ready for s in self._slots)
            )

    def topology(self) -> dict:
        with self._lock:
            return {
                "num_shards":      self.num_shards,
                "model":           self.model,
                "workers": [
                    {
                        "shard_idx":  s.shard_idx,
                        "host":       s.host,
                        "input_port": s.input_port,
                        "ready":      s.ready,
                    }
                    for s in sorted(self._slots, key=lambda x: x.shard_idx)
                ],
                "registered":      len(self._slots),
                "complete":        (
                    len(self._slots) >= self.num_shards
                    and all(s.ready for s in self._slots)
                ),
                "slots_remaining": max(0, self.num_shards - len(self._slots)),
            }


# ── FastAPI app ───────────────────────────────────────────────────────────────


def _make_app(coord: NetworkCoordinator):                # type: ignore[return]
    """Build and return a FastAPI app wrapping *coord*."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError:
        raise ImportError("fastapi is required: pip install fastapi uvicorn")

    app = FastAPI(title="LankaMind Coordinator", version="0.6.0")

    @app.get("/health")
    def health():
        topo = coord.topology()
        return {
            "status":     "ok",
            "registered": topo["registered"],
            "num_shards": topo["num_shards"],
            "ready":      sum(1 for w in topo["workers"] if w["ready"]),
            "complete":   topo["complete"],
        }

    @app.post("/api/register")
    def register(body: dict):
        host   = body.get("host", "localhost")
        result = coord.register(host)
        if result.get("error") == "network_full":
            raise HTTPException(status_code=409, detail="Network is full — all shards assigned")
        return result

    @app.get("/api/topology")
    def topology():
        return coord.topology()

    @app.get("/api/config/{shard_idx}")
    def config(shard_idx: int):
        cfg = coord.get_config(shard_idx)
        if cfg is None:
            raise HTTPException(
                status_code=425,
                detail="Topology not complete — waiting for more workers to join",
            )
        return cfg

    @app.post("/api/ready/{shard_idx}")
    def ready(shard_idx: int):
        ok = coord.mark_ready(shard_idx)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Shard {shard_idx} not registered")
        return {"ok": True, "shard_idx": shard_idx}

    return app


# ── Background runner ─────────────────────────────────────────────────────────


def _local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def run_coordinator(
    num_shards:    int,
    model:         str,
    port:          int  = DEFAULT_COORD_PORT,
    bind_host:     str  = "0.0.0.0",
    result_host:   str  = "",
    announce_mdns: bool = True,
) -> NetworkCoordinator:
    """
    Start the coordinator REST API in a background thread.

    Returns the live ``NetworkCoordinator`` so the caller can inspect
    topology, wait for ``coord.all_ready``, etc.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required: pip install uvicorn")

    local_ip    = _local_ip()
    result_host = result_host or local_ip
    coord       = NetworkCoordinator(num_shards=num_shards, model=model, result_host=result_host)
    app         = _make_app(coord)

    # mDNS announcement — other devices discover us automatically
    if announce_mdns:
        try:
            from network.discovery import announce
            announce(
                service_name=f"lankamind-coord",
                port=port,
                role="coordinator",
                host_ip=local_ip,
                extra_props={
                    "model":      model,
                    "num_shards": str(num_shards),
                    "coord_port": str(port),
                    "version":    "0.6.0",
                },
            )
            log.info("mDNS announced coordinator at %s:%d", local_ip, port)
        except Exception as exc:
            log.warning("mDNS announce failed (non-fatal): %s", exc)

    # Run uvicorn in a daemon thread
    cfg    = uvicorn.Config(app, host=bind_host, port=port, log_level="warning")
    server = uvicorn.Server(cfg)

    t = threading.Thread(target=server.run, daemon=True, name="LankaMindCoord")
    t.start()

    # Wait until the server is up (max 10 s)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            import requests as _req
            _req.get(f"http://localhost:{port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)

    log.info(
        "Coordinator ready  http://%s:%d  (model=%s, shards=%d)",
        local_ip, port, model, num_shards,
    )
    return coord

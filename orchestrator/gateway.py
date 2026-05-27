"""
orchestrator/gateway.py
------------------------
The Gateway is the "brain" of the LankaMind network.

Ports (defaults):
  5700  PULL  — receives heartbeat JSON messages from workers
  5701  REP   — answers client chain-discovery requests
  9090  HTTP  — Prometheus /metrics endpoint (if prometheus-client is installed)

Message formats
----------------
Worker → Gateway (heartbeat, every 5 s):
  {
    "type": "heartbeat",
    "worker_id": "<uuid>",
    "shard_idx": 0,
    "num_shards": 3,
    "model_name": "gpt2",
    "host": "localhost",
    "port": 5500,
    "requests_processed": 42
  }

Client → Gateway (REQ):
  { "type": "get_chain", "model_name": "gpt2", "num_shards": 3 }
  OR
  { "type": "status" }

Gateway → Client (REP, success):
  { "status": "ok", "chain": [ {worker_dict}, ... ] }

Gateway → Client (REP, error):
  { "status": "error", "message": "..." }

Run directly:
  python -m orchestrator.gateway [--heartbeat-port N] [--client-port N]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

import zmq

from orchestrator.health_checker import HealthChecker
from orchestrator.registry import NodeRegistry
from orchestrator.scheduler import NotEnoughWorkersError, Scheduler

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[Gateway] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

HEARTBEAT_PORT = 5700
CLIENT_PORT    = 5701
METRICS_PORT   = 9090


class Gateway:
    def __init__(
        self,
        heartbeat_port: int = HEARTBEAT_PORT,
        client_port:    int = CLIENT_PORT,
        metrics_port:   int = METRICS_PORT,
    ):
        self.registry        = NodeRegistry()
        self.scheduler       = Scheduler(self.registry)
        self.health_checker  = HealthChecker(self.registry)
        self.heartbeat_port  = heartbeat_port
        self.client_port     = client_port
        self.metrics_port    = metrics_port
        self._requests_total = 0
        self._stop           = threading.Event()

    # ── Heartbeat receiver ────────────────────────────────────────────────────

    def _heartbeat_loop(self, ctx: zmq.Context) -> None:
        sock = ctx.socket(zmq.PULL)
        sock.RCVTIMEO = 1_000   # ms; lets us check _stop every second
        sock.bind(f"tcp://*:{self.heartbeat_port}")
        log.info("Heartbeat receiver on port %d", self.heartbeat_port)

        while not self._stop.is_set():
            try:
                msg = sock.recv_json()
            except zmq.Again:
                continue
            except Exception as exc:
                log.warning("Heartbeat recv error: %s", exc)
                continue

            if msg.get("type") == "heartbeat":
                self.registry.register(
                    worker_id=msg["worker_id"],
                    shard_idx=msg["shard_idx"],
                    num_shards=msg["num_shards"],
                    model_name=msg["model_name"],
                    host=msg.get("host", "localhost"),
                    port=msg["port"],
                    requests_processed=msg.get("requests_processed", 0),
                )
        sock.close()

    # ── Client request handler ────────────────────────────────────────────────

    def _client_loop(self, ctx: zmq.Context) -> None:
        sock = ctx.socket(zmq.REP)
        sock.RCVTIMEO = 1_000
        sock.bind(f"tcp://*:{self.client_port}")
        log.info("Client directory on port %d", self.client_port)

        while not self._stop.is_set():
            try:
                req = sock.recv_json()
            except zmq.Again:
                continue
            except Exception as exc:
                log.warning("Client recv error: %s", exc)
                continue

            req_type = req.get("type", "")

            if req_type == "get_chain":
                try:
                    chain = self.scheduler.build_pipeline(
                        req["model_name"], req["num_shards"]
                    )
                    self._requests_total += 1
                    log.info(
                        "Chain built for '%s' (%d shards): %s",
                        req["model_name"],
                        req["num_shards"],
                        [w.address for w in chain],
                    )
                    sock.send_json({
                        "status": "ok",
                        "chain": [w.to_dict() for w in chain],
                    })
                except NotEnoughWorkersError as exc:
                    sock.send_json({"status": "error", "message": str(exc)})
                except Exception as exc:
                    sock.send_json({"status": "error", "message": str(exc)})

            elif req_type == "status":
                sock.send_json({
                    "status":          "ok",
                    "workers":         self.registry.snapshot(),
                    "requests_total":  self._requests_total,
                })

            else:
                sock.send_json({
                    "status":  "error",
                    "message": f"Unknown request type: {req_type!r}",
                })

        sock.close()

    # ── Prometheus metrics ────────────────────────────────────────────────────

    def _start_metrics(self) -> None:
        try:
            from prometheus_client import Counter, Gauge, start_http_server

            active_g  = Gauge("lankamind_active_workers",   "Registered workers in pool")
            healthy_g = Gauge("lankamind_healthy_workers",  "Healthy workers in pool")
            reqs_c    = Counter("lankamind_requests_total", "Chain requests served by gateway")

            def _update_loop() -> None:
                while not self._stop.is_set():
                    snap      = self.registry.snapshot()
                    active_g.set(len(snap))
                    healthy_g.set(sum(1 for w in snap if w["is_healthy"]))
                    reqs_c._value.set(self._requests_total)   # type: ignore[attr-defined]
                    time.sleep(5)

            start_http_server(self.metrics_port)
            threading.Thread(target=_update_loop, daemon=True).start()
            log.info("Prometheus metrics on http://localhost:%d/metrics", self.metrics_port)
        except ImportError:
            log.info("prometheus-client not installed; metrics endpoint disabled")
        except Exception as exc:
            log.warning("Could not start metrics: %s", exc)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        ctx = zmq.Context()

        self.health_checker.start()
        self._start_metrics()

        hb_thread = threading.Thread(
            target=self._heartbeat_loop, args=(ctx,), daemon=True, name="HB"
        )
        cl_thread = threading.Thread(
            target=self._client_loop, args=(ctx,), daemon=True, name="Client"
        )
        hb_thread.start()
        cl_thread.start()

        log.info("Gateway ready. Press Ctrl-C to stop.")

        def _shutdown(*_: object) -> None:
            log.info("Shutting down gateway…")
            self._stop.set()
            self.health_checker.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown)

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            _shutdown()


# ── CLI entry ─────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="LankaMind Gateway — orchestrates the worker pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--heartbeat-port", type=int, default=HEARTBEAT_PORT)
    p.add_argument("--client-port",    type=int, default=CLIENT_PORT)
    p.add_argument("--metrics-port",   type=int, default=METRICS_PORT)
    args = p.parse_args()
    Gateway(args.heartbeat_port, args.client_port, args.metrics_port).run()


if __name__ == "__main__":
    main()

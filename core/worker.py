"""
core/worker.py
--------------
Worker process: loads one ModelShard and loops forever, processing requests.

PHASE 1 behaviour (unchanged):
  Load shard → bind PULL socket → process activations → PUSH to next hop.

PHASE 2 additions (optional — only active if --gateway-address is given):
  • Heartbeat thread: PUSHes a JSON status message to the gateway every 5 s
    so the gateway knows this worker is alive and what shard it serves.
  • Ping responder: binds a REP socket on (input_port + 100) and replies
    "pong" to every "ping" — used by the gateway's HealthChecker.

All Phase 2 flags are OPTIONAL.  Omitting --gateway-address runs Phase 1 mode.

Run:
    python -m core.worker --shard-idx 0 --num-shards 3 --model gpt2 \
           --input-port 5500 --output-address tcp://localhost:5501

With gateway:
    python -m core.worker --shard-idx 0 --num-shards 3 --model gpt2 \
           --input-port 5500 --output-address tcp://localhost:5501 \
           --gateway-address tcp://localhost:5700 --host 192.168.1.10
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import signal
import sys
import tempfile
import threading
import time
import uuid

import zmq

from core.model_shard import ModelShard
from transport.zmq_transport import (
    WorkerInputSocket,
    WorkerOutputSocket,
    deserialize_tensor,
)

logging.basicConfig(
    level=logging.INFO,
    format="[Worker %(shard_idx)s] %(message)s",
)

HEARTBEAT_INTERVAL_S = 5.0


# ── Readiness sentinel ────────────────────────────────────────────────────────


def _ready_file(shard_idx: int) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"lankamind_worker_{shard_idx}.ready"


def _cleanup(shard_idx: int) -> None:
    try:
        _ready_file(shard_idx).unlink(missing_ok=True)
    except Exception:
        pass


# ── Phase 2: heartbeat thread ─────────────────────────────────────────────────


def _heartbeat_loop(
    stop_event:      threading.Event,
    gateway_address: str,
    worker_id:       str,
    shard_idx:       int,
    num_shards:      int,
    model_name:      str,
    host:            str,
    input_port:      int,
    request_counter: list,   # [0] = mutable integer shared with main loop
) -> None:
    """
    Continuously PUSH heartbeat messages to the gateway.
    Runs in a daemon thread — exits when stop_event is set.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(gateway_address)

    while not stop_event.wait(HEARTBEAT_INTERVAL_S):
        try:
            sock.send_json({
                "type":                "heartbeat",
                "worker_id":           worker_id,
                "shard_idx":           shard_idx,
                "num_shards":          num_shards,
                "model_name":          model_name,
                "host":                host,
                "port":                input_port,
                "requests_processed":  request_counter[0],
            })
        except Exception:
            pass   # don't crash the heartbeat thread

    sock.close()
    ctx.term()


# ── Phase 2: ping responder thread ───────────────────────────────────────────


def _ping_responder_loop(
    stop_event: threading.Event,
    ping_port:  int,
) -> None:
    """
    Bind a REP socket on ping_port and reply "pong" to every "ping".
    The gateway's HealthChecker measures round-trip time to this socket.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.RCVTIMEO = 500   # ms
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.bind(f"tcp://*:{ping_port}")
    except zmq.ZMQError as exc:
        # Non-fatal: if the ping port is already in use, just skip
        logging.getLogger(__name__).warning(
            "Could not bind ping port %d: %s", ping_port, exc,
            extra={"shard_idx": "?"},
        )
        ctx.term()
        return

    while not stop_event.wait(0):
        try:
            msg = sock.recv_string()
            if msg == "ping":
                sock.send_string("pong")
        except zmq.Again:
            pass
        except Exception:
            pass

    sock.close()
    ctx.term()


# ── Main worker logic ─────────────────────────────────────────────────────────


def run_worker(
    shard_idx:       int,
    num_shards:      int,
    model_name:      str,
    input_port:      int,
    output_address:  str,
    gateway_address: str | None = None,
    worker_id:       str | None = None,
    host:            str        = "localhost",
) -> None:
    """
    Load the model shard and start the inference loop.
    Never returns under normal operation.
    """
    log   = logging.getLogger(__name__)
    extra = {"shard_idx": shard_idx}
    wid   = worker_id or uuid.uuid4().hex[:12]

    # ── 1. Load model shard ───────────────────────────────────────────────────
    log.info("Loading shard %d/%d from '%s' …", shard_idx, num_shards - 1, model_name, extra=extra)
    shard = ModelShard(model_name, shard_idx, num_shards)
    log.info(
        "Loaded layers %s  is_first=%s  is_last=%s",
        shard.layer_range, shard.is_first, shard.is_last,
        extra=extra,
    )

    # ── 2. ZMQ inference sockets ─────────────────────────────────────────────
    ctx     = zmq.Context()
    in_sock = WorkerInputSocket(ctx, input_port)
    out_sock = WorkerOutputSocket(ctx, output_address)
    log.info("Inference: port %d → %s", input_port, output_address, extra=extra)

    # ── 3. Phase 2 background threads ────────────────────────────────────────
    stop_event      = threading.Event()
    request_counter = [0]   # mutable counter shared with inference loop

    if gateway_address:
        ping_port = input_port + 100
        hb_thread = threading.Thread(
            target=_heartbeat_loop,
            kwargs=dict(
                stop_event=stop_event,
                gateway_address=gateway_address,
                worker_id=wid,
                shard_idx=shard_idx,
                num_shards=num_shards,
                model_name=model_name,
                host=host,
                input_port=input_port,
                request_counter=request_counter,
            ),
            daemon=True,
            name=f"HB-shard{shard_idx}",
        )
        ping_thread = threading.Thread(
            target=_ping_responder_loop,
            kwargs=dict(stop_event=stop_event, ping_port=ping_port),
            daemon=True,
            name=f"Ping-shard{shard_idx}",
        )
        hb_thread.start()
        ping_thread.start()
        log.info(
            "Heartbeat → %s  |  Ping responder on port %d",
            gateway_address, ping_port,
            extra=extra,
        )

    # ── 4. Signal readiness ───────────────────────────────────────────────────
    _ready_file(shard_idx).touch()
    log.info("READY  worker_id=%s", wid, extra=extra)

    # ── 5. Graceful shutdown ──────────────────────────────────────────────────
    def _shutdown(*_: object) -> None:
        log.info("Shutting down.", extra=extra)
        stop_event.set()
        _cleanup(shard_idx)
        in_sock.close()
        out_sock.close()
        ctx.term()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)

    # ── 6. Inference loop ─────────────────────────────────────────────────────
    try:
        while True:
            header, tensor_bytes = in_sock.recv()

            if shard.is_first:
                output = shard(input_ids=deserialize_tensor(tensor_bytes))
            else:
                output = shard(hidden_states=deserialize_tensor(tensor_bytes))

            request_counter[0] += 1

            if shard.is_last:
                next_token_id = int(output[0, -1, :].argmax().item())
                header["next_token_id"] = next_token_id
                out_sock.send(header, tensor=None)
            else:
                out_sock.send(header, tensor=output)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt.", extra=extra)
    finally:
        stop_event.set()
        _cleanup(shard_idx)
        in_sock.close()
        out_sock.close()
        ctx.term()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="LankaMind worker — holds one model shard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--shard-idx",       type=int, required=True)
    p.add_argument("--num-shards",      type=int, required=True)
    p.add_argument("--model",           type=str, default="gpt2")
    p.add_argument("--input-port",      type=int, required=True)
    p.add_argument("--output-address",  type=str, required=True)
    # Phase 2 flags (optional)
    p.add_argument("--gateway-address", type=str, default=None,
                   help="Gateway heartbeat address (e.g. tcp://localhost:5700)")
    p.add_argument("--worker-id",       type=str, default=None,
                   help="Stable worker ID (auto-generated UUID if omitted)")
    p.add_argument("--host",            type=str, default="localhost",
                   help="Public host/IP reported to the gateway")
    args = p.parse_args()

    run_worker(
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        model_name=args.model,
        input_port=args.input_port,
        output_address=args.output_address,
        gateway_address=args.gateway_address,
        worker_id=args.worker_id,
        host=args.host,
    )


if __name__ == "__main__":
    main()

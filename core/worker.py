"""
core/worker.py
--------------
Worker process: loads one ModelShard and loops forever, processing requests.

Each worker is a standalone Python process.  The launch script
(scripts/launch_workers.py) spawns N of them with different --shard-idx
values.  Together they form the inference pipeline.

HOW A REQUEST FLOWS THROUGH A WORKER
======================================
  1.  Receive a two-frame ZMQ message from the upstream worker (or client):
        Frame 0: JSON header  { request_id, generation_step }
        Frame 1: serialised tensor  (input_ids for shard 0; hidden_states otherwise)

  2.  Deserialise the tensor and run the forward pass through our model slice.

  3a. If we are NOT the last shard: send [header, new_hidden_states] to the
      next worker's PULL socket.

  3b. If we ARE the last shard: we have logits.  Take the argmax of the
      last token position to get the next predicted token ID.  Send
      [header+{"next_token_id": N}, b""] back to the client.

READY SIGNALLING
=================
Loading a 500 MB model takes a few seconds.  When ready, each worker writes
an empty "sentinel" file to the system temp directory so the launch script
and integration tests know all shards are up.

Run this file directly:
    python -m core.worker --shard-idx 0 --num-shards 3 \\
           --model gpt2 --input-port 5500 \\
           --output-address tcp://localhost:5501
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import signal
import sys
import tempfile

import torch
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


def _ready_file(shard_idx: int) -> pathlib.Path:
    """Path to the sentinel file we touch when we finish loading."""
    return pathlib.Path(tempfile.gettempdir()) / f"lankamind_worker_{shard_idx}.ready"


def _cleanup(shard_idx: int) -> None:
    """Remove the ready sentinel on exit so stale files don't confuse restarts."""
    try:
        _ready_file(shard_idx).unlink(missing_ok=True)
    except Exception:
        pass


# ── Main worker loop ──────────────────────────────────────────────────────────


def run_worker(
    shard_idx: int,
    num_shards: int,
    model_name: str,
    input_port: int,
    output_address: str,
) -> None:
    """
    Load the model shard and start the request-processing loop.

    This function never returns under normal operation — it loops until the
    process is killed (SIGTERM / Ctrl-C).
    """
    log = logging.getLogger(__name__)
    extra = {"shard_idx": shard_idx}

    # ── 1. Load model shard ───────────────────────────────────────────────────
    log.info("Loading shard %d/%d from '%s' …", shard_idx, num_shards - 1, model_name, extra=extra)
    shard = ModelShard(model_name, shard_idx, num_shards)
    shard.eval()
    log.info(
        "Loaded — transformer blocks %s, is_first=%s, is_last=%s",
        shard.layer_range,
        shard.is_first,
        shard.is_last,
        extra=extra,
    )

    # ── 2. Set up ZMQ sockets ─────────────────────────────────────────────────
    ctx = zmq.Context()

    # Input: bind a PULL socket so upstream can PUSH to us
    in_sock = WorkerInputSocket(ctx, input_port)
    log.info("Listening on port %d", input_port, extra=extra)

    # Output: connect a PUSH socket to the next hop
    out_sock = WorkerOutputSocket(ctx, output_address)
    log.info("Forwarding to %s", output_address, extra=extra)

    # ── 3. Signal readiness ───────────────────────────────────────────────────
    _ready_file(shard_idx).touch()
    log.info("READY — waiting for requests …", extra=extra)

    # ── 4. Graceful shutdown on SIGTERM ───────────────────────────────────────
    def _handle_sigterm(*_):
        log.info("Received SIGTERM — shutting down.", extra=extra)
        _cleanup(shard_idx)
        in_sock.close()
        out_sock.close()
        ctx.term()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # ── 5. Main loop ──────────────────────────────────────────────────────────
    try:
        while True:
            header, tensor_bytes = in_sock.recv()

            req_id = header.get("request_id", "?")
            step = header.get("generation_step", 0)

            # Deserialise and run forward pass
            if shard.is_first:
                input_ids = deserialize_tensor(tensor_bytes)
                log.debug(
                    "req=%s step=%d  input_ids shape=%s",
                    req_id, step, tuple(input_ids.shape),
                    extra=extra,
                )
                output = shard(input_ids=input_ids)
            else:
                hidden_states = deserialize_tensor(tensor_bytes)
                log.debug(
                    "req=%s step=%d  hidden_states shape=%s",
                    req_id, step, tuple(hidden_states.shape),
                    extra=extra,
                )
                output = shard(hidden_states=hidden_states)

            # Forward result downstream
            if shard.is_last:
                # output is logits [batch, seq_len, vocab_size]
                # Greedy: take the token with the highest score at the last position
                next_token_id = int(output[0, -1, :].argmax().item())
                header["next_token_id"] = next_token_id
                log.debug("req=%s step=%d  next_token=%d", req_id, step, next_token_id, extra=extra)
                out_sock.send(header, tensor=None)  # empty tensor frame — result is in header
            else:
                out_sock.send(header, tensor=output)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt — shutting down.", extra=extra)
    finally:
        _cleanup(shard_idx)
        in_sock.close()
        out_sock.close()
        ctx.term()


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LankaMind worker — holds one shard of the model pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shard-idx",      type=int, required=True,  help="Index of this shard (0-based)")
    parser.add_argument("--num-shards",     type=int, required=True,  help="Total number of shards in the pipeline")
    parser.add_argument("--model",          type=str, default="gpt2", help="HuggingFace model name or local path")
    parser.add_argument("--input-port",     type=int, required=True,  help="TCP port this worker listens on")
    parser.add_argument("--output-address", type=str, required=True,  help="Address of the next hop (tcp://host:port)")

    args = parser.parse_args()

    run_worker(
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        model_name=args.model,
        input_port=args.input_port,
        output_address=args.output_address,
    )


if __name__ == "__main__":
    main()

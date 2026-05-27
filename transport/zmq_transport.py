"""
transport/zmq_transport.py
--------------------------
Thin ZeroMQ helpers for passing PyTorch tensors between worker processes.

WHY ZEROMQ?
===========
ZeroMQ (zmq) is a battle-tested messaging library that works like a
supercharged socket.  Key properties we rely on:

• No broker/server required — workers connect directly to each other.
• PUSH / PULL pattern: one sender, one receiver, messages queue up safely
  if the receiver is temporarily busy.
• Automatic reconnect — if a worker restarts, ZMQ retries the connection
  without losing queued messages (up to the high-water mark).
• Cross-platform: works identically on Windows, Mac, Linux.

When we move to real P2P in Phase 3, we replace these socket classes with
libp2p stream wrappers and change nothing else in the codebase.

MESSAGE FORMAT (two ZMQ frames)
================================
  Frame 0 — UTF-8 JSON header  { request_id, generation_step, ... }
  Frame 1 — raw bytes          torch.save(tensor) output, or b"" if none

TOPOLOGY (Phase 1 — single machine)
=====================================
                          ┌──────────┐
   CLI client             │ Worker 0 │ port 5500  PULL ← client PUSH
   PUSH ──────────────→   │  Shard 0 │            PUSH → 5501
                          └──────────┘
                          ┌──────────┐
                          │ Worker 1 │ port 5501  PULL ← Worker 0
                          │  Shard 1 │            PUSH → 5502
                          └──────────┘
                          ┌──────────┐
                          │ Worker 2 │ port 5502  PULL ← Worker 1
                          │  Shard 2 │            PUSH → 5599
                          └──────────┘
   CLI client PULL  ←─────────────────────────────────────────────
   (binds port 5599)
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Tuple

import torch
import zmq


# ── Tensor serialisation ──────────────────────────────────────────────────────


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """
    Encode a PyTorch tensor to raw bytes.

    We use torch.save() which is essentially pickle — fast and handles every
    dtype (float32, float16, int64, …) without any extra work.

    Trade-off: pickle is Python-only. Phase 3 will switch to a
    language-neutral format (e.g. safetensors or flatbuffers) so non-Python
    workers (mobile, Rust) can participate.
    """
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """Decode bytes produced by serialize_tensor() back into a tensor."""
    if not data:
        raise ValueError("Tried to deserialise an empty byte string as a tensor.")
    buf = io.BytesIO(data)
    # weights_only=False because we serialise full tensor objects.
    # Phase 3 note: switch to signed / verified messages for security.
    return torch.load(buf, weights_only=False)  # noqa: S614


# ── Socket wrappers ───────────────────────────────────────────────────────────


class WorkerInputSocket:
    """
    PULL socket that a worker **binds** to receive incoming messages.

    Workers are stable long-running processes → they bind (i.e. they own
    the address).  Clients and upstream workers connect to them.
    """

    def __init__(self, ctx: zmq.Context, port: int) -> None:
        self._sock: zmq.Socket = ctx.socket(zmq.PULL)
        self._sock.set_hwm(500)  # queue up to 500 messages before blocking
        self._sock.bind(f"tcp://*:{port}")

    def recv(self) -> Tuple[Dict[str, Any], bytes]:
        """
        Block until a message arrives.

        Returns
        -------
        header      : dict parsed from JSON frame 0
        tensor_bytes: raw bytes from frame 1 (may be b"" for final hop)
        """
        header_bytes, tensor_bytes = self._sock.recv_multipart()
        header = json.loads(header_bytes.decode("utf-8"))
        return header, tensor_bytes

    def close(self) -> None:
        self._sock.close()


class WorkerOutputSocket:
    """
    PUSH socket that a worker **connects** to send messages downstream.

    The downstream endpoint (next worker or client) is the stable binder;
    this socket is the ephemeral connector.
    """

    def __init__(self, ctx: zmq.Context, address: str) -> None:
        self._sock: zmq.Socket = ctx.socket(zmq.PUSH)
        self._sock.set_hwm(500)
        self._sock.connect(address)

    def send(
        self,
        header: Dict[str, Any],
        tensor: torch.Tensor | None = None,
    ) -> None:
        """
        Send a two-frame multipart message.

        Parameters
        ----------
        header : dict   — request metadata (will be JSON-encoded)
        tensor : Tensor | None — the activation or None for the final hop
        """
        header_bytes = json.dumps(header).encode("utf-8")
        tensor_bytes = serialize_tensor(tensor) if tensor is not None else b""
        self._sock.send_multipart([header_bytes, tensor_bytes])

    def close(self) -> None:
        self._sock.close()

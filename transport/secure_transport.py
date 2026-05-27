"""
transport/secure_transport.py
------------------------------
ZMQ CURVE-encrypted socket factories.

CURVE encryption uses Elliptic-Curve Diffie-Hellman (via libsodium / Curve25519).
Every connection is mutually authenticated:
  • The server has a long-term keypair.
  • The client knows the server's public key and has its own ephemeral keypair.
  • libzmq performs the handshake automatically after setting the socket options.

Classes
-------
SecureWorkerInputSocket   — encrypted PULL socket (server role)
SecureWorkerOutputSocket  — encrypted PUSH socket (client role)

Both are drop-in replacements for their unencrypted counterparts in
transport.zmq_transport; they have identical send/recv interfaces.

Graceful-fallback pattern (used by worker.py and client.py)
-----------------------------------------------------------
If zmq.has("curve") is False (rare — requires libsodium), the factory
functions fall back to plain unencrypted sockets and log a warning.

Usage
-----
    from transport.secure_transport import make_secure_pull, make_secure_push

    # Server (worker):
    sock = make_secure_pull(ctx, port=5500, server_secret_z85="...", server_public_z85="...")

    # Client:
    sock = make_secure_push(ctx, address="tcp://host:5500", server_public_z85="...")
"""

from __future__ import annotations

import io
import json
import logging
import struct
from typing import Optional, Tuple

import torch
import zmq

log = logging.getLogger(__name__)

_CURVE_AVAILABLE: bool = zmq.has("curve")
if not _CURVE_AVAILABLE:
    log.warning(
        "libzmq was compiled without libsodium — CURVE encryption is not available. "
        "Falling back to unencrypted sockets."
    )


# ── Tensor serialisation (identical to zmq_transport) ────────────────────────

def serialize_tensor(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(data), weights_only=False)


# ── Low-level socket factories ────────────────────────────────────────────────

def _apply_server_curve(sock: zmq.Socket, server_public_z85: str, server_secret_z85: str) -> None:
    """Configure a ZMQ socket as a CURVE *server*."""
    if not _CURVE_AVAILABLE:
        return
    sock.curve_server = True
    sock.curve_publickey = server_public_z85.encode("ascii")
    sock.curve_secretkey = server_secret_z85.encode("ascii")


def _apply_client_curve(sock: zmq.Socket, server_public_z85: str) -> None:
    """Configure a ZMQ socket as a CURVE *client* (uses ephemeral keypair)."""
    if not _CURVE_AVAILABLE:
        return
    client_pub, client_sec = zmq.curve_keypair()
    sock.curve_publickey = client_pub
    sock.curve_secretkey = client_sec
    sock.curve_serverkey = server_public_z85.encode("ascii")


def make_secure_pull(
    ctx: zmq.Context,
    port: int,
    server_public_z85: str,
    server_secret_z85: str,
) -> zmq.Socket:
    """
    Create an encrypted PULL socket that binds to *port*.

    Parameters
    ----------
    server_public_z85, server_secret_z85 : Z85-encoded server keypair strings.
    """
    sock = ctx.socket(zmq.PULL)
    _apply_server_curve(sock, server_public_z85, server_secret_z85)
    sock.bind(f"tcp://*:{port}")
    return sock


def make_secure_push(
    ctx: zmq.Context,
    address: str,
    server_public_z85: str,
) -> zmq.Socket:
    """
    Create an encrypted PUSH socket that connects to *address*.

    The client generates an ephemeral keypair automatically; only the server's
    public key needs to be known in advance (trust-on-first-use model).
    """
    sock = ctx.socket(zmq.PUSH)
    _apply_client_curve(sock, server_public_z85)
    sock.connect(address)
    return sock


# ── High-level socket wrappers (drop-in for zmq_transport) ───────────────────

class SecureWorkerInputSocket:
    """
    Encrypted PULL socket for a worker (server role).

    Interface identical to transport.zmq_transport.WorkerInputSocket.
    """

    def __init__(
        self,
        ctx: zmq.Context,
        port: int,
        server_public_z85: str,
        server_secret_z85: str,
    ) -> None:
        self._sock = make_secure_pull(ctx, port, server_public_z85, server_secret_z85)

    def recv(self) -> Tuple[dict, bytes]:
        """Receive (header_dict, tensor_bytes)."""
        header_bytes, tensor_bytes = self._sock.recv_multipart()
        return json.loads(header_bytes.decode("utf-8")), tensor_bytes

    def close(self) -> None:
        self._sock.close()


class SecureWorkerOutputSocket:
    """
    Encrypted PUSH socket for a worker or client (client role).

    Interface identical to transport.zmq_transport.WorkerOutputSocket.
    """

    def __init__(
        self,
        ctx: zmq.Context,
        address: str,
        server_public_z85: str,
    ) -> None:
        self._sock = make_secure_push(ctx, address, server_public_z85)

    def send(self, header: dict, tensor: Optional[torch.Tensor]) -> None:
        """Send (header_dict, tensor_bytes) multipart message."""
        header_bytes = json.dumps(header).encode("utf-8")
        tensor_bytes = serialize_tensor(tensor) if tensor is not None else b""
        self._sock.send_multipart([header_bytes, tensor_bytes])

    def close(self) -> None:
        self._sock.close()


# ── Utility ───────────────────────────────────────────────────────────────────

def curve_available() -> bool:
    """Return True if libzmq was compiled with libsodium (CURVE support)."""
    return _CURVE_AVAILABLE

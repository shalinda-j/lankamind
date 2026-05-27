"""
tests/unit/test_secure_transport.py
-------------------------------------
Unit tests for transport.secure_transport.

Tests run in two modes:
  • CURVE available (libsodium present):  full encryption roundtrip tested.
  • CURVE not available:                 curve_available() returns False,
                                         module still importable, sockets work unencrypted.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch
import zmq

from transport.secure_transport import (
    curve_available,
    deserialize_tensor,
    serialize_tensor,
)


# ── Serialisation (not curve-dependent) ──────────────────────────────────────

class TestSerialisation:
    def test_roundtrip_float(self):
        t = torch.randn(2, 4)
        assert torch.allclose(t, deserialize_tensor(serialize_tensor(t)))

    def test_roundtrip_long(self):
        t = torch.tensor([[1, 2, 3]], dtype=torch.long)
        result = deserialize_tensor(serialize_tensor(t))
        assert torch.equal(t, result)

    def test_dtype_preserved(self):
        t = torch.zeros(1, 5, dtype=torch.float16)
        assert deserialize_tensor(serialize_tensor(t)).dtype == torch.float16


# ── curve_available() ─────────────────────────────────────────────────────────

class TestCurveAvailable:
    def test_returns_bool(self):
        assert isinstance(curve_available(), bool)

    def test_matches_zmq(self):
        assert curve_available() == zmq.has("curve")


# ── Encrypted roundtrip (skipped when CURVE unavailable) ─────────────────────

@pytest.mark.skipif(not zmq.has("curve"), reason="CURVE not available")
class TestSecureSockets:
    """
    Spin up a SecureWorkerInputSocket (PULL/server) + SecureWorkerOutputSocket
    (PUSH/client) in threads and verify a multipart message arrives intact.
    """

    _PORT = 15_700   # use a high port unlikely to conflict

    def _server_thread(self, pub: str, sec: str, received: list, ready: threading.Event) -> None:
        from transport.secure_transport import SecureWorkerInputSocket
        ctx = zmq.Context()
        sock = SecureWorkerInputSocket(ctx, self._PORT, pub, sec)
        ready.set()
        try:
            header, tensor_bytes = sock.recv()
            received.append((header, tensor_bytes))
        finally:
            sock.close()
            ctx.term()

    def test_encrypted_roundtrip(self):
        from network.keypair import generate
        from transport.secure_transport import SecureWorkerOutputSocket

        pub, sec = generate()
        received: list = []
        ready = threading.Event()

        server = threading.Thread(
            target=self._server_thread,
            args=(pub, sec, received, ready),
            daemon=True,
        )
        server.start()
        ready.wait(timeout=5)
        time.sleep(0.1)   # let the socket bind

        ctx = zmq.Context()
        client = SecureWorkerOutputSocket(ctx, f"tcp://localhost:{self._PORT}", pub)
        tensor = torch.tensor([[1, 2, 3]], dtype=torch.long)
        client.send({"step": 0, "next_token_id": 42}, tensor=tensor)
        client.close()

        server.join(timeout=5)
        ctx.term()

        assert len(received) == 1
        header, tensor_bytes = received[0]
        assert header["step"] == 0
        assert header["next_token_id"] == 42
        recovered = deserialize_tensor(tensor_bytes)
        assert torch.equal(recovered, tensor)

    def test_make_secure_pull_binds(self):
        from network.keypair import generate
        from transport.secure_transport import make_secure_pull

        pub, sec = generate()
        ctx = zmq.Context()
        sock = make_secure_pull(ctx, self._PORT + 1, pub, sec)
        assert sock is not None
        sock.close()
        ctx.term()

    def test_make_secure_push_connects(self):
        """Push socket should connect without error (server doesn't need to be up)."""
        from network.keypair import generate
        from transport.secure_transport import make_secure_push

        pub, _ = generate()
        ctx = zmq.Context()
        sock = make_secure_push(ctx, f"tcp://localhost:{self._PORT + 2}", pub)
        assert sock is not None
        sock.close()
        ctx.term()

"""
tests/unit/test_zmq_transport.py
---------------------------------
Unit tests for tensor serialisation helpers in transport/zmq_transport.py.

These tests do NOT open any network sockets — they only test the
serialize_tensor / deserialize_tensor round-trip.

Run:
    pytest tests/unit/test_zmq_transport.py -v
"""

import pytest
import torch

from transport.zmq_transport import deserialize_tensor, serialize_tensor


class TestTensorSerialization:

    def test_roundtrip_float32(self):
        t = torch.randn(2, 16, 128)
        recovered = deserialize_tensor(serialize_tensor(t))
        assert torch.allclose(t, recovered), "Float32 tensor not recovered correctly"

    def test_roundtrip_int64(self):
        t = torch.randint(0, 1000, (1, 20))
        recovered = deserialize_tensor(serialize_tensor(t))
        assert torch.equal(t, recovered), "Int64 tensor not recovered correctly"

    def test_dtype_float16_preserved(self):
        t = torch.randn(4, 8).half()
        recovered = deserialize_tensor(serialize_tensor(t))
        assert recovered.dtype == torch.float16, "float16 dtype not preserved"
        assert torch.allclose(t, recovered)

    def test_shape_preserved(self):
        shapes = [(1,), (3, 4), (1, 50, 768), (2, 10, 10, 10)]
        for shape in shapes:
            t = torch.randn(*shape)
            recovered = deserialize_tensor(serialize_tensor(t))
            assert recovered.shape == t.shape, f"Shape mismatch for {shape}"

    def test_single_element_tensor(self):
        t = torch.tensor([[42]])
        recovered = deserialize_tensor(serialize_tensor(t))
        assert recovered.item() == 42

    def test_all_zeros(self):
        t = torch.zeros(5, 5)
        recovered = deserialize_tensor(serialize_tensor(t))
        assert torch.equal(t, recovered)

    def test_serialised_bytes_are_bytes(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = serialize_tensor(t)
        assert isinstance(result, bytes), "serialize_tensor should return bytes"
        assert len(result) > 0, "serialize_tensor should not return empty bytes"

    def test_empty_bytes_raises(self):
        with pytest.raises((Exception,)):
            deserialize_tensor(b"")

    def test_large_activation_tensor(self):
        """Simulate a realistic hidden-state tensor from GPT-2 small."""
        # batch=1, seq_len=100, hidden_dim=768  →  ~300 KB
        t = torch.randn(1, 100, 768)
        recovered = deserialize_tensor(serialize_tensor(t))
        assert torch.allclose(t, recovered, atol=1e-6)

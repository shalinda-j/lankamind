"""
tests/unit/test_keypair.py
---------------------------
Unit tests for network.keypair — Curve25519 keypair management.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import zmq

from network.keypair import generate, get_or_create, load, save


CURVE_AVAILABLE = zmq.has("curve")
pytestmark = pytest.mark.skipif(
    not CURVE_AVAILABLE,
    reason="libzmq not compiled with libsodium — CURVE unavailable",
)


class TestGenerate:
    def test_returns_tuple_of_two_strings(self):
        pub, sec = generate()
        assert isinstance(pub, str)
        assert isinstance(sec, str)

    def test_z85_length_40_chars(self):
        pub, sec = generate()
        # Z85-encoded 32-byte key → 40 ASCII chars
        assert len(pub) == 40
        assert len(sec) == 40

    def test_z85_only_printable_ascii(self):
        pub, sec = generate()
        for char in pub + sec:
            assert 0x20 <= ord(char) <= 0x7E

    def test_each_call_produces_unique_keys(self):
        pub1, sec1 = generate()
        pub2, sec2 = generate()
        assert pub1 != pub2
        assert sec1 != sec2

    def test_public_and_secret_differ(self):
        pub, sec = generate()
        assert pub != sec


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: pathlib.Path):
        path = tmp_path / "node.key"
        pub, sec = generate()
        save(path, pub, sec)
        loaded_pub, loaded_sec = load(path)
        assert loaded_pub == pub
        assert loaded_sec == sec

    def test_save_creates_parent_dirs(self, tmp_path: pathlib.Path):
        path = tmp_path / "a" / "b" / "node.key"
        pub, sec = generate()
        save(path, pub, sec)
        assert path.exists()

    def test_save_writes_valid_json(self, tmp_path: pathlib.Path):
        path = tmp_path / "node.key"
        pub, sec = generate()
        save(path, pub, sec)
        data = json.loads(path.read_text())
        assert "public" in data
        assert "secret" in data

    def test_load_missing_file_raises(self, tmp_path: pathlib.Path):
        path = tmp_path / "does_not_exist.key"
        with pytest.raises(FileNotFoundError):
            load(path)

    def test_load_bad_json_raises(self, tmp_path: pathlib.Path):
        path = tmp_path / "bad.key"
        path.write_text("not valid json")
        with pytest.raises(Exception):  # json.JSONDecodeError
            load(path)

    def test_load_missing_key_raises(self, tmp_path: pathlib.Path):
        path = tmp_path / "missing.key"
        path.write_text(json.dumps({"public": "x"}))  # no 'secret'
        with pytest.raises(KeyError):
            load(path)


class TestGetOrCreate:
    def test_creates_file_if_missing(self, tmp_path: pathlib.Path):
        path = tmp_path / "node.key"
        assert not path.exists()
        pub, sec = get_or_create(path)
        assert path.exists()
        assert len(pub) == 40

    def test_returns_same_keys_on_second_call(self, tmp_path: pathlib.Path):
        path = tmp_path / "node.key"
        pub1, sec1 = get_or_create(path)
        pub2, sec2 = get_or_create(path)
        assert pub1 == pub2
        assert sec1 == sec2

    def test_does_not_overwrite_existing(self, tmp_path: pathlib.Path):
        path = tmp_path / "node.key"
        pub_orig, sec_orig = generate()
        save(path, pub_orig, sec_orig)
        pub_loaded, sec_loaded = get_or_create(path)
        assert pub_loaded == pub_orig
        assert sec_loaded == sec_orig

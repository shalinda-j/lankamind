"""
network/keypair.py
------------------
Curve25519 keypair management for ZMQ CURVE encryption.

ZMQ CURVE uses Curve25519 keys (32-byte public + 32-byte secret), stored as
Z85-encoded strings (40 printable ASCII chars each).

Functions
---------
generate()          → (public_z85: str, secret_z85: str)
save(path, pub, sec)  → writes JSON file to *path*
load(path)          → (public_z85: str, secret_z85: str)
get_or_create(path) → load if exists, else generate + save

Example
-------
    from network.keypair import get_or_create
    pub, sec = get_or_create(Path.home() / ".lankamind" / "node.key")
"""

from __future__ import annotations

import json
import pathlib
from typing import Tuple

import zmq


KeyPair = Tuple[str, str]   # (public_z85, secret_z85)


def generate() -> KeyPair:
    """
    Generate a fresh Curve25519 keypair.

    Returns
    -------
    (public_z85, secret_z85) — both are 40-char Z85-encoded strings.
    """
    public_z85, secret_z85 = zmq.curve_keypair()
    # zmq returns bytes; decode to str for JSON-friendly storage
    return public_z85.decode("ascii"), secret_z85.decode("ascii")


def save(path: pathlib.Path, public_z85: str, secret_z85: str) -> None:
    """
    Persist a keypair to *path* as a JSON file.

    The directory is created if it does not exist.
    File permissions are set to 0o600 (owner read/write only).
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"public": public_z85, "secret": secret_z85}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)   # best-effort on Windows
    except NotImplementedError:
        pass


def load(path: pathlib.Path) -> KeyPair:
    """
    Load a keypair from *path*.

    Raises
    ------
    FileNotFoundError  — if the file does not exist.
    KeyError           — if the JSON is missing 'public' or 'secret'.
    """
    path = pathlib.Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["public"], data["secret"]


def get_or_create(path: pathlib.Path) -> KeyPair:
    """
    Return the keypair at *path* if it exists, otherwise generate and save one.

    This is the primary convenience function for production use:

        pub, sec = get_or_create(Path.home() / ".lankamind" / "node.key")
    """
    path = pathlib.Path(path)
    if path.exists():
        return load(path)
    pub, sec = generate()
    save(path, pub, sec)
    return pub, sec

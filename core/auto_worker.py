"""
core/auto_worker.py
-------------------
Auto-joining worker: discovers the network coordinator and self-configures.

A device runs ONE command::

    lankamind join                              # auto-discover on LAN
    lankamind join --coordinator http://IP:5800 # explicit coordinator
    lankamind join --model distilgpt2           # override model

The device then:
  1. Discovers the coordinator (mDNS or explicit URL)
  2. Registers its LAN IP → gets a shard index assigned
  3. Waits until all shards have registered (topology is complete)
  4. Downloads its portion of the model
  5. Starts the inference worker loop (runs forever)
  6. Signals the coordinator that it is ready

No manual --shard-idx, --input-port, or --output-address needed.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Optional

log = logging.getLogger(__name__)

DISCOVER_TIMEOUT   = 30   # s  — mDNS browse window
TOPOLOGY_TIMEOUT   = 300  # s  — wait for all shards to register
DEFAULT_COORD_PORT = 5800


# ── Helpers ───────────────────────────────────────────────────────────────────


def _local_ip() -> str:
    """Get this machine's LAN IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _discover_coordinator(timeout: float = DISCOVER_TIMEOUT) -> str | None:
    """
    Browse the LAN via mDNS for a LankaMind coordinator.
    Returns the coordinator HTTP URL or None if not found.
    """
    try:
        from network.discovery import browse_once
        services = browse_once(timeout=min(timeout, 10))
        for svc in services:
            if svc.get("role") == "coordinator":
                host = svc.get("host") or svc.get("address", "")
                port = int(svc.get("coord_port") or svc.get("port") or DEFAULT_COORD_PORT)
                if host:
                    url = f"http://{host}:{port}"
                    log.info("mDNS: found coordinator at %s", url)
                    return url
    except Exception as exc:
        log.debug("mDNS discovery error: %s", exc)
    return None


# ── Main auto-join entry point ────────────────────────────────────────────────


def auto_join(
    coordinator_url: Optional[str] = None,
    model:           Optional[str] = None,
    host_ip:         Optional[str] = None,
    gateway_address: Optional[str] = None,
) -> None:
    """
    Discover the network, register this device, and start a worker.
    This function **blocks forever** (the worker inference loop runs until killed).

    Parameters
    ----------
    coordinator_url : str, optional
        If given, skip mDNS discovery and use this URL directly.
        Example: ``"http://192.168.1.10:5800"``
    model : str, optional
        Override the model name the coordinator suggests.
    host_ip : str, optional
        Override the LAN IP reported to the coordinator.
        Useful on multi-homed machines.
    gateway_address : str, optional
        Optional gateway heartbeat address (Phase 2 feature).
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required: pip install requests")

    my_ip = host_ip or _local_ip()

    print(f"\n  {'='*56}")
    print(f"  LankaMind — Auto-Join")
    print(f"  {'='*56}")
    print(f"  This device : {my_ip}")

    # ── Step 1: discover coordinator ─────────────────────────────────────────
    if coordinator_url is None:
        print("  Looking for coordinator on LAN (mDNS)...", end="", flush=True)
        coordinator_url = _discover_coordinator()
        if coordinator_url is None:
            print(" not found.\n")
            raise RuntimeError(
                "No LankaMind coordinator found on this network.\n\n"
                "  On the main PC, run:\n"
                "    lankamind start --model distilgpt2 --shards 3\n\n"
                "  Or specify coordinator directly:\n"
                "    lankamind join --coordinator http://<IP>:5800"
            )
        print(f" found!\n  Coordinator : {coordinator_url}")
    else:
        print(f"  Coordinator : {coordinator_url}")

    # ── Step 2: register ──────────────────────────────────────────────────────
    print("  Registering...", end="", flush=True)
    try:
        resp = requests.post(
            f"{coordinator_url}/api/register",
            json={"host": my_ip},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to register with coordinator: {exc}") from exc

    reg        = resp.json()
    shard_idx  = reg["shard_idx"]
    num_shards = reg["num_shards"]
    model_name = model or reg.get("model", "distilgpt2")
    remaining  = reg.get("slots_remaining", 0)

    print(f" assigned shard {shard_idx + 1}/{num_shards}")
    print(f"  Model       : {model_name}")
    print(f"  Input port  : {reg['input_port']}")

    if remaining > 0:
        print(f"\n  Waiting for {remaining} more device(s) to join...")

    # ── Step 3: wait for full topology ────────────────────────────────────────
    cfg      = None
    deadline = time.monotonic() + TOPOLOGY_TIMEOUT
    dots     = 0

    while time.monotonic() < deadline:
        try:
            r = requests.get(
                f"{coordinator_url}/api/config/{shard_idx}",
                timeout=5,
            )
            if r.status_code == 200:
                cfg = r.json()
                break
            # 425 = not all shards registered yet
        except Exception:
            pass

        dots += 1
        print(f"\r  Topology building{'.' * (dots % 4):3s}  ", end="", flush=True)
        time.sleep(3)

    if cfg is None:
        raise RuntimeError(
            f"Timed out waiting for all {num_shards} shards to join "
            f"(waited {TOPOLOGY_TIMEOUT}s). "
            "Other devices may not have run 'lankamind join' yet."
        )

    print(f"\r  {'='*56}")
    print(f"  All {num_shards} shards registered — starting worker")
    print(f"  Route  : port {cfg['input_port']} → {cfg['output_address']}")
    print(f"  {'='*56}\n")

    # ── Step 4 + 5: load model and start inference loop ───────────────────────
    # Import here so the startup print above appears before slow model loading
    from core.worker import run_worker as _run_worker

    # Signal the coordinator we're ready AFTER the model loads.
    # run_worker() calls _ready_file().touch() internally, but the coordinator
    # doesn't know about local files on other machines — we hook in via a
    # background thread that polls the ready file and then calls /api/ready.
    import pathlib, tempfile, threading

    def _notify_ready() -> None:
        """Poll the local ready-file and tell coordinator once it appears."""
        ready_path = pathlib.Path(tempfile.gettempdir()) / f"lankamind_worker_{shard_idx}.ready"
        t0 = time.monotonic()
        while time.monotonic() - t0 < 300:
            if ready_path.exists():
                try:
                    requests.post(
                        f"{coordinator_url}/api/ready/{shard_idx}",
                        timeout=5,
                    )
                    log.info("Notified coordinator: shard %d ready", shard_idx)
                except Exception:
                    pass
                return
            time.sleep(1)

    threading.Thread(target=_notify_ready, daemon=True, name="ReadyNotifier").start()

    _run_worker(
        shard_idx=shard_idx,
        num_shards=num_shards,
        model_name=model_name,
        input_port=cfg["input_port"],
        output_address=cfg["output_address"],
        gateway_address=gateway_address,
        host=my_ip,
    )

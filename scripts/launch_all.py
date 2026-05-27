"""
scripts/launch_all.py
----------------------
ONE COMMAND to start the entire LankaMind stack on a single machine:

  1. Gateway (port 5700/5701)
  2. N Worker shards (ports 5500…)
  3. REST API + Mobile Web UI (port 8000)
  4. mDNS announcement → phones on same Wi-Fi open http://lankamind.local:8000

Usage
-----
    python scripts/launch_all.py
    python scripts/launch_all.py --shards 3 --model gpt2 --port 8000
    lankamind serve                      # same thing via unified CLI

After startup you will see:
  ┌─────────────────────────────────────────────────────┐
  │  LankaMind is READY                                 │
  │  Open on this machine : http://localhost:8000       │
  │  Open on your phone   : http://192.168.1.5:8000     │
  │  mDNS (any browser)   : http://lankamind.local:8000 │
  └─────────────────────────────────────────────────────┘

Android / Termux workers
------------------------
To add an Android phone as a worker node, install Termux and run:

    pkg install python
    pip install lankamind
    lankamind node --gateway tcp://<THIS-PC-IP>:5700 --host <PHONE-IP>
"""

from __future__ import annotations

import argparse
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import threading

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
BASE_PORT = 5500
RESULT_PORT = 5599
LOAD_TIMEOUT_SECONDS = 180


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _ready_file(shard_idx: int) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"lankamind_worker_{shard_idx}.ready"


def _cleanup_ready(num_shards: int) -> None:
    for i in range(num_shards):
        _ready_file(i).unlink(missing_ok=True)


def _announce_mdns(port: int) -> tuple:
    """Announce API via mDNS. Returns (zc, info) or (None, None)."""
    try:
        from network.discovery import announce
        return announce("LankaMind API", port=port, role="api")
    except Exception:
        return None, None


def _print_banner(ip: str, port: int) -> None:
    url_local = f"http://localhost:{port}"
    url_lan   = f"http://{ip}:{port}"
    url_mdns  = f"http://lankamind.local:{port}"
    width = 57
    print()
    print("┌" + "─" * width + "┐")
    print(f"│{'  ✅  LankaMind is READY':^{width}}│")
    print("├" + "─" * width + "┤")
    print(f"│  Open on this machine : {url_local:<{width-26}}│")
    print(f"│  Open on your phone   : {url_lan:<{width-26}}│")
    print(f"│  mDNS (auto-discover) : {url_mdns:<{width-26}}│")
    print("├" + "─" * width + "┤")
    print(f"│  API docs             : {url_local + '/docs':<{width-26}}│")
    print(f"│  Wi-Fi QR code        : {url_local + '/v1/network-info':<{width-26}}│")
    print("└" + "─" * width + "┘")
    print()
    print("  Any device on this Wi-Fi: open the phone URL in a browser.")
    print("  No app installation needed — works in Safari, Chrome, etc.")
    print()
    print("  Press Ctrl-C to stop everything.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the complete LankaMind stack (gateway + workers + API).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shards",  type=int, default=3,     help="Number of worker shards")
    parser.add_argument("--model",   type=str, default="gpt2", help="Model name")
    parser.add_argument("--port",    type=int, default=8000,  help="API / web UI port")
    parser.add_argument("--no-mdns", action="store_true",     help="Disable mDNS announcement")
    args = parser.parse_args()

    LOGS_DIR.mkdir(exist_ok=True)
    _cleanup_ready(args.shards)

    ip = _local_ip()

    env = os.environ.copy()
    pp  = str(PROJECT_ROOT)
    env["PYTHONPATH"] = f"{pp}{os.pathsep}{env.get('PYTHONPATH','')}"
    env["LANKAMIND_BASE_PORT"]   = str(BASE_PORT)
    env["LANKAMIND_RESULT_PORT"] = str(RESULT_PORT)
    env["LANKAMIND_GATEWAY"]     = "tcp://localhost:5701"
    env["LANKAMIND_API_PORT"]    = str(args.port)

    processes: list[subprocess.Popen] = []

    # ── 1. Gateway ────────────────────────────────────────────────────────────
    gw_log = open(LOGS_DIR / "gateway.log", "w", encoding="utf-8")
    gw = subprocess.Popen(
        [sys.executable, "-m", "orchestrator.gateway",
         "--heartbeat-port", "5700",
         "--client-port",    "5701"],
        cwd=PROJECT_ROOT, env=env,
        stdout=gw_log, stderr=gw_log,
    )
    processes.append(gw)
    print(f"  [1/3] Gateway started (PID {gw.pid})")
    time.sleep(1.5)   # give gateway time to bind ports

    # ── 2. Workers ────────────────────────────────────────────────────────────
    for i in range(args.shards):
        input_port = BASE_PORT + i
        if i < args.shards - 1:
            output_address = f"tcp://localhost:{BASE_PORT + i + 1}"
        else:
            output_address = f"tcp://localhost:{RESULT_PORT}"

        wlog = open(LOGS_DIR / f"worker_{i}.log", "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "core.worker",
             "--shard-idx",      str(i),
             "--num-shards",     str(args.shards),
             "--model",          args.model,
             "--input-port",     str(input_port),
             "--output-address", output_address,
             "--gateway-address","tcp://localhost:5700",
             "--host",           ip],
            cwd=PROJECT_ROOT, env=env,
            stdout=wlog, stderr=wlog,
        )
        processes.append(proc)
        print(f"  [2/3] Worker {i} started (PID {proc.pid}, port {input_port})")

    # ── 3. Wait for workers to load ───────────────────────────────────────────
    print(f"\n  Waiting for {args.shards} model shards to load …")
    print("  (First run downloads GPT-2 weights ~500 MB — be patient)\n")
    start = time.monotonic()
    while time.monotonic() - start < LOAD_TIMEOUT_SECONDS:
        flags = [_ready_file(i).exists() for i in range(args.shards)]
        if all(flags):
            break
        for i, p in enumerate(processes):
            if p.poll() is not None:
                print(f"\n  ✗ Process {i} exited (code {p.returncode}).")
                print(f"  Check logs/...")
                _terminate_all(processes)
                sys.exit(1)
        time.sleep(2)
        print(f"  {sum(flags)}/{args.shards} shards ready …", end="\r", flush=True)
    else:
        print("\n  ✗ Timeout waiting for workers.")
        _terminate_all(processes)
        sys.exit(1)

    elapsed = time.monotonic() - start
    print(f"\n  ✓ All {args.shards} workers ready in {elapsed:.1f}s")

    # ── 4. mDNS announcement ──────────────────────────────────────────────────
    zc, zc_info = (None, None)
    if not args.no_mdns:
        print("  Announcing on local network via mDNS …")
        zc, zc_info = _announce_mdns(args.port)
        if zc:
            print(f"  ✓ mDNS: lankamind.local:{args.port}")
        else:
            print("  ⚠ mDNS unavailable — use LAN IP instead")

    # ── 5. API server ─────────────────────────────────────────────────────────
    api_log = open(LOGS_DIR / "api.log", "w", encoding="utf-8")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.server:app",
         "--host", "0.0.0.0",
         "--port", str(args.port),
         "--log-level", "warning"],
        cwd=PROJECT_ROOT, env=env,
        stdout=api_log, stderr=api_log,
    )
    processes.append(api_proc)
    time.sleep(1.5)

    print(f"  [3/3] API server started (PID {api_proc.pid}, port {args.port})")

    _print_banner(ip, args.port)

    # ── 6. Wait / Ctrl-C ─────────────────────────────────────────────────────
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down …")
    finally:
        if zc and zc_info:
            try:
                zc.unregister_service(zc_info)
                zc.close()
            except Exception:
                pass
        _cleanup_ready(args.shards)
        _terminate_all(processes)
        print("All services stopped.")


def _terminate_all(processes: list) -> None:
    import subprocess as sp
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + 5
    for p in processes:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            p.wait(timeout=remaining)
        except sp.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    main()

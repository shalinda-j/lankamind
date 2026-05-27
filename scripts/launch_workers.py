"""
scripts/launch_workers.py
--------------------------
Spawn N worker processes, each holding one shard of the model.

Usage
-----
    python scripts/launch_workers.py               # 3 shards of GPT-2
    python scripts/launch_workers.py --shards 4
    python scripts/launch_workers.py --shards 2 --model gpt2-medium

What happens
------------
  1. Removes any leftover "ready" sentinel files from previous runs.
  2. Spawns N subprocesses, one per shard (core.worker).
  3. Polls the temp directory for sentinel files that each worker writes
     when it has finished loading its model shard.
  4. Prints the exact command to run the CLI client once all workers are up.
  5. Waits; forwards Ctrl-C to all workers on exit.

Worker logs go to:  logs/worker_<N>.log
PIDs are saved to:  .worker_pids  (read by stop_workers.py)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
PID_FILE = PROJECT_ROOT / ".worker_pids"
BASE_PORT = 5500
RESULT_PORT = 5599
LOAD_TIMEOUT_SECONDS = 180   # GPT-2 can take ~30 s on first download


def ready_file(shard_idx: int) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"lankamind_worker_{shard_idx}.ready"


def cleanup_ready_files(num_shards: int) -> None:
    for i in range(num_shards):
        ready_file(i).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch LankaMind worker processes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shards", type=int, default=3, help="Number of worker shards")
    parser.add_argument("--model",  type=str, default="gpt2", help="HuggingFace model name")
    args = parser.parse_args()

    num_shards: int = args.shards
    model_name: str = args.model

    LOGS_DIR.mkdir(exist_ok=True)
    cleanup_ready_files(num_shards)

    # ── Prepare environment: add project root to PYTHONPATH ───────────────────
    env = os.environ.copy()
    python_path = str(PROJECT_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{python_path}{os.pathsep}{existing}" if existing else python_path

    # ── Spawn each worker ─────────────────────────────────────────────────────
    processes: list[subprocess.Popen] = []
    pids: list[int] = []

    print(f"Starting {num_shards} workers for model '{model_name}' …\n")

    for i in range(num_shards):
        input_port = BASE_PORT + i

        # Last worker sends to the client's result port; others go to next worker
        if i < num_shards - 1:
            output_address = f"tcp://localhost:{BASE_PORT + i + 1}"
        else:
            output_address = f"tcp://localhost:{RESULT_PORT}"

        cmd = [
            sys.executable, "-m", "core.worker",
            "--shard-idx",      str(i),
            "--num-shards",     str(num_shards),
            "--model",          model_name,
            "--input-port",     str(input_port),
            "--output-address", output_address,
        ]

        log_path = LOGS_DIR / f"worker_{i}.log"
        log_file = open(log_path, "w", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )
        processes.append(proc)
        pids.append(proc.pid)

        print(f"  ✓  Worker {i}  (shard {i}/{num_shards-1})  PID {proc.pid}  "
              f"port {input_port}  →  log: logs/worker_{i}.log")

    # Save PIDs so stop_workers.py can find them
    PID_FILE.write_text("\n".join(str(p) for p in pids) + "\n")

    # ── Wait for all shards to finish loading ─────────────────────────────────
    print(f"\nWaiting for model shards to load (timeout {LOAD_TIMEOUT_SECONDS}s) …")
    print("(Each worker downloads ~500 MB on the very first run — this is normal)\n")

    start = time.monotonic()
    dots = 0

    while time.monotonic() - start < LOAD_TIMEOUT_SECONDS:
        ready_flags = [ready_file(i).exists() for i in range(num_shards)]

        if all(ready_flags):
            break

        # Check whether any worker crashed
        for i, proc in enumerate(processes):
            if proc.poll() is not None:
                print(f"\n✗ Worker {i} exited unexpectedly (code {proc.returncode}).")
                print(f"  Check logs/worker_{i}.log for details.")
                _terminate_all(processes)
                sys.exit(1)

        time.sleep(2)
        dots += 1
        ready_count = sum(ready_flags)
        print(f"  {ready_count}/{num_shards} shards ready …", end="\r", flush=True)

    else:
        print("\n✗ Timed out waiting for workers.")
        print("  Check logs/worker_*.log for error details.")
        _terminate_all(processes)
        sys.exit(1)

    elapsed = time.monotonic() - start
    print(f"\n{'─'*60}")
    print(f"  ✓  All {num_shards} workers ready in {elapsed:.1f}s")
    print(f"{'─'*60}")
    print()
    print("Run the client in a NEW terminal window:")
    print()
    print(f'    python cli/client.py "Sri Lanka is a beautiful island that"')
    print()
    print("Press Ctrl-C here to stop all workers.")
    print()

    # ── Keep running; forward Ctrl-C to all workers ───────────────────────────
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down workers …")
        _terminate_all(processes)
        cleanup_ready_files(num_shards)
        PID_FILE.unlink(missing_ok=True)
        print("All workers stopped.")


def _terminate_all(processes: list[subprocess.Popen]) -> None:
    for proc in processes:
        try:
            proc.terminate()
        except Exception:
            pass
    # Give them 5 s to exit gracefully, then kill
    deadline = time.monotonic() + 5
    for proc in processes:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()

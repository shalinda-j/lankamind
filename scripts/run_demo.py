"""
scripts/run_demo.py
--------------------
Self-contained end-to-end demo: starts workers, runs inference, prints output.
No gateway needed — direct mode only.

Quick start (no download needed — tiny model already cached):
    python scripts/run_demo.py --model sshleifer/tiny-gpt2

Full quality (downloads ~500 MB on first run):
    python scripts/run_demo.py --model gpt2
    python scripts/run_demo.py --model distilgpt2

Custom prompt:
    python scripts/run_demo.py --model sshleifer/tiny-gpt2 --prompt "Once upon a time in Colombo" --tokens 40
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import signal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_PORT    = 5500
RESULT_PORT  = 5599
NUM_SHARDS   = 3
TIMEOUT      = 600   # 10 minutes — covers first-run download on slow connections

# Force UTF-8 output so Unicode symbols work on Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def ready_file(i: int) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"lankamind_worker_{i}.ready"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The history of Sri Lanka spans")
    ap.add_argument("--tokens", type=int, default=50)
    ap.add_argument("--model",  default="gpt2")
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"

    # Clear old ready files
    for i in range(NUM_SHARDS):
        ready_file(i).unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print("  LankaMind — Live Inference Demo")
    print(f"{'='*60}")
    print(f"  Model  : {args.model}")
    print(f"  Shards : {NUM_SHARDS}")
    print(f"  Prompt : {args.prompt}")
    print(f"{'='*60}\n")

    procs: list[subprocess.Popen] = []

    # ── Spawn workers ────────────────────────────────────────────────
    LOGS = PROJECT_ROOT / "logs"
    LOGS.mkdir(exist_ok=True)

    for i in range(NUM_SHARDS):
        in_port  = BASE_PORT + i
        out_addr = (
            f"tcp://localhost:{BASE_PORT + i + 1}"
            if i < NUM_SHARDS - 1
            else f"tcp://localhost:{RESULT_PORT}"
        )
        log_f = open(LOGS / f"worker_{i}.log", "w")
        p = subprocess.Popen(
            [sys.executable, "-m", "core.worker",
             "--shard-idx",      str(i),
             "--num-shards",     str(NUM_SHARDS),
             "--model",          args.model,
             "--input-port",     str(in_port),
             "--output-address", out_addr],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_f,
            stderr=log_f,
        )
        procs.append(p)
        print(f"  Worker {i} starting (PID {p.pid}, port {in_port}) …")

    # ── Wait for all shards ready ────────────────────────────────────
    print("\n  Waiting for model to load …")
    if args.model in ("gpt2", "openai-community/gpt2"):
        print("  (First run downloads GPT-2 weights ~500 MB — this is normal)")
    elif args.model in ("distilgpt2",):
        print("  (First run downloads distilgpt2 weights ~82 MB — this is normal)")
    else:
        print("  (Tiny model — should load in seconds from local cache)")
    print()

    start = time.monotonic()
    while time.monotonic() - start < TIMEOUT:
        flags = [ready_file(i).exists() for i in range(NUM_SHARDS)]
        n = sum(flags)
        print(f"  {n}/{NUM_SHARDS} shards ready …", end="\r", flush=True)

        if all(flags):
            elapsed = time.monotonic() - start
            print(f"\n  ✓ All {NUM_SHARDS} shards loaded in {elapsed:.1f}s\n")
            break

        for i, p in enumerate(procs):
            if p.poll() is not None:
                print(f"\n  ✗ Worker {i} crashed (code {p.returncode})")
                print(f"  Check logs/worker_{i}.log")
                _kill(procs)
                sys.exit(1)
        time.sleep(2)
    else:
        print("\n  ✗ Timed out waiting for workers.")
        _kill(procs)
        sys.exit(1)

    # ── Run inference ────────────────────────────────────────────────
    try:
        # Import inside so PYTHONPATH is already set
        sys.path.insert(0, str(PROJECT_ROOT))
        from cli.client import run_client
        result = run_client(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            model=args.model,
            workers=NUM_SHARDS,
            base_port=BASE_PORT,
            result_port=RESULT_PORT,
        )
        print(f"\n{'='*60}")
        print(f"  ✅ Inference complete!")
        print(f"  Generated: {result[:200]}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _kill(procs)
        for i in range(NUM_SHARDS):
            ready_file(i).unlink(missing_ok=True)
        print("  Workers stopped.")


def _kill(procs: list) -> None:
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + 5
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    main()

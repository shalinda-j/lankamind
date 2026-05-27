"""
scripts/stop_workers.py
------------------------
Stop all running LankaMind workers and clean up sentinel files.

Usage
-----
    python scripts/stop_workers.py

This reads PIDs from .worker_pids (written by launch_workers.py) and
sends SIGTERM (Unix) / taskkill (Windows) to each process.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import time


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PID_FILE = PROJECT_ROOT / ".worker_pids"


def ready_file(shard_idx: int) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"lankamind_worker_{shard_idx}.ready"


def main() -> None:
    if not PID_FILE.exists():
        print("No .worker_pids file found — workers may not be running.")
        # Still clean up any stale ready files
        for path in pathlib.Path(tempfile.gettempdir()).glob("lankamind_worker_*.ready"):
            path.unlink(missing_ok=True)
        return

    pids = [int(line.strip()) for line in PID_FILE.read_text().splitlines() if line.strip()]

    if not pids:
        print("PID file is empty.")
        PID_FILE.unlink(missing_ok=True)
        return

    print(f"Stopping {len(pids)} worker(s): PIDs {pids}")

    for pid in pids:
        try:
            if sys.platform == "win32":
                result = subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"  ✓ Stopped PID {pid}")
                else:
                    print(f"  ○ PID {pid} already gone ({result.stderr.strip()})")
            else:
                import signal as _signal
                os.kill(pid, _signal.SIGTERM)
                print(f"  ✓ Sent SIGTERM to PID {pid}")
        except (ProcessLookupError, PermissionError) as exc:
            print(f"  ○ PID {pid}: {exc}")

    # Clean up files
    PID_FILE.unlink(missing_ok=True)
    for path in pathlib.Path(tempfile.gettempdir()).glob("lankamind_worker_*.ready"):
        path.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()

"""
scripts/launch_gateway.py
--------------------------
Start the LankaMind Gateway process.

The gateway is the orchestration hub:
  • Receives worker heartbeats  (port 5700)
  • Answers chain-discovery requests from clients  (port 5701)
  • Exposes Prometheus metrics  (port 9090)

Usage:
    python scripts/launch_gateway.py
    python scripts/launch_gateway.py --heartbeat-port 5700 --client-port 5701

Start Order (recommended):
  1. python scripts/launch_gateway.py
  2. python scripts/launch_workers.py --shards 3 --gateway tcp://localhost:5700
  3. python cli/client.py "Your prompt" --gateway tcp://localhost:5701
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the LankaMind Gateway",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--heartbeat-port", type=int, default=5700)
    parser.add_argument("--client-port",    type=int, default=5701)
    parser.add_argument("--metrics-port",   type=int, default=9090)
    args = parser.parse_args()

    env = os.environ.copy()
    pp  = str(PROJECT_ROOT)
    env["PYTHONPATH"] = f"{pp}{os.pathsep}{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "orchestrator.gateway",
        "--heartbeat-port", str(args.heartbeat_port),
        "--client-port",    str(args.client_port),
        "--metrics-port",   str(args.metrics_port),
    ]

    print(f"Starting gateway…")
    print(f"  Heartbeat port : {args.heartbeat_port}")
    print(f"  Client port    : {args.client_port}")
    print(f"  Metrics port   : {args.metrics_port}")
    print()
    print("Now start workers:")
    print(f"  python scripts/launch_workers.py --shards 3 "
          f"--gateway tcp://localhost:{args.heartbeat_port}")
    print()

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    except KeyboardInterrupt:
        print("\nGateway stopped.")


if __name__ == "__main__":
    main()

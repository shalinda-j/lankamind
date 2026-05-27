"""
scripts/launch_bootstrap.py
----------------------------
Start the LankaMind Bootstrap / peer-discovery node.

Usage:
    python scripts/launch_bootstrap.py
    python scripts/launch_bootstrap.py --port 6000

The bootstrap node is optional.  Workers can still operate without it if
they connect directly via the gateway.  The bootstrap node helps new nodes
discover the gateway's address in fully distributed deployments.
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
        description="Launch the LankaMind Bootstrap node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=6000, help="ZMQ REP port")
    args = parser.parse_args()

    env = os.environ.copy()
    pp  = str(PROJECT_ROOT)
    env["PYTHONPATH"] = f"{pp}{os.pathsep}{env.get('PYTHONPATH', '')}"

    cmd = [sys.executable, "-m", "network.bootstrap", "--port", str(args.port)]

    print(f"Starting bootstrap node on port {args.port} …")
    print("Workers can discover peers by connecting to:")
    print(f"  tcp://localhost:{args.port}")
    print()

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    except KeyboardInterrupt:
        print("\nBootstrap node stopped.")


if __name__ == "__main__":
    main()

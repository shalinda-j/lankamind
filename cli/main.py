"""
cli/main.py
-----------
Unified LankaMind CLI entry point.

Commands
--------
lankamind chat       — interactive text completion (REPL)
lankamind complete   — single-shot completion (same as cli/client.py)
lankamind node       — start a worker node
lankamind gateway    — start the gateway
lankamind bootstrap  — start the peer-discovery bootstrap node
lankamind api        — start the REST API server
lankamind status     — show gateway + worker status
lankamind balance    — show reward ledger balance for a worker
lankamind keys       — print this node's public key

Install:
    pip install -e .
    lankamind --help
"""

from __future__ import annotations

import pathlib
import sys

import click


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_project_root() -> None:
    """Ensure the project root is on sys.path (for editable installs)."""
    root = pathlib.Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.5.0", prog_name="lankamind")
def cli() -> None:
    """LankaMind — distributed LLM inference for Sri Lanka."""
    _add_project_root()


# ── complete ──────────────────────────────────────────────────────────────────

@cli.command("complete")
@click.argument("prompt")
@click.option("--max-tokens",  default=60,    show_default=True, help="Max tokens to generate")
@click.option("--model",       default="gpt2", show_default=True, help="Model name")
@click.option("--workers",     default=3,     show_default=True, help="Number of pipeline workers")
@click.option("--base-port",   default=5500,  show_default=True, help="Worker 0 port")
@click.option("--result-port", default=5599,  show_default=True, help="Client result port")
@click.option("--gateway",     default=None,                     help="Gateway address (tcp://...)")
@click.option("--encrypt",     is_flag=True,  default=False,     help="Enable CURVE encryption")
@click.option("--server-key",  default=None,                     help="Worker 0 Z85 public key")
def complete(
    prompt, max_tokens, model, workers, base_port, result_port, gateway, encrypt, server_key
) -> None:
    """Generate a single text completion."""
    from cli.client import run_client
    run_client(
        prompt=prompt,
        max_new_tokens=max_tokens,
        model=model,
        workers=workers,
        base_port=base_port,
        result_port=result_port,
        gateway_address=gateway,
        encrypt=encrypt,
        server_key=server_key,
    )


# ── chat ──────────────────────────────────────────────────────────────────────

@cli.command("chat")
@click.option("--model",       default="gpt2", show_default=True, help="Model name")
@click.option("--workers",     default=3,     show_default=True, help="Number of pipeline workers")
@click.option("--max-tokens",  default=60,    show_default=True, help="Max tokens per turn")
@click.option("--gateway",     default=None,                     help="Gateway address")
@click.option("--base-port",   default=5500,  show_default=True)
@click.option("--result-port", default=5599,  show_default=True)
def chat(model, workers, max_tokens, gateway, base_port, result_port) -> None:
    """Interactive text completion REPL (Ctrl-C or 'exit' to quit)."""
    from cli.client import run_client

    click.echo(f"LankaMind chat ({model}, {workers} workers). Type 'exit' to quit.\n")
    while True:
        try:
            prompt = click.prompt("You")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nBye!")
            break
        if prompt.strip().lower() in ("exit", "quit", "q"):
            break
        run_client(
            prompt=prompt,
            max_new_tokens=max_tokens,
            model=model,
            workers=workers,
            base_port=base_port,
            result_port=result_port,
            gateway_address=gateway,
        )
        click.echo()


# ── node ──────────────────────────────────────────────────────────────────────

@cli.command("node")
@click.option("--shards",      default=3,     show_default=True, help="Number of shards")
@click.option("--model",       default="gpt2", show_default=True, help="Model name")
@click.option("--gateway",     default=None,                     help="Gateway heartbeat address")
@click.option("--host",        default="localhost", show_default=True, help="Advertised host/IP")
def node(shards, model, gateway, host) -> None:
    """Start worker nodes (spawns N subprocesses)."""
    import subprocess, os

    root = pathlib.Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable, str(root / "scripts" / "launch_workers.py"),
        "--shards", str(shards),
        "--model", model,
    ]
    if gateway:
        cmd += ["--gateway", gateway, "--host", host]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=root, env=env)


# ── gateway ───────────────────────────────────────────────────────────────────

@cli.command("gateway")
@click.option("--heartbeat-port", default=5700, show_default=True)
@click.option("--client-port",    default=5701, show_default=True)
@click.option("--metrics-port",   default=9090, show_default=True)
def gateway_cmd(heartbeat_port, client_port, metrics_port) -> None:
    """Start the LankaMind gateway / orchestrator."""
    from orchestrator.gateway import Gateway
    Gateway(heartbeat_port, client_port, metrics_port).run()


# ── bootstrap ─────────────────────────────────────────────────────────────────

@cli.command("bootstrap")
@click.option("--port", default=6000, show_default=True, help="Bootstrap node port")
def bootstrap(port) -> None:
    """Start the peer-discovery bootstrap node."""
    from network.bootstrap import BootstrapNode
    BootstrapNode(port=port).run()


# ── api ───────────────────────────────────────────────────────────────────────

@cli.command("api")
@click.option("--host",    default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port",    default=8000,      show_default=True, help="HTTP port")
@click.option("--reload",  is_flag=True,      default=False,     help="Auto-reload (dev mode)")
@click.option("--gateway", default=None,      help="Gateway address to set LANKAMIND_GATEWAY env var")
def api_cmd(host, port, reload, gateway) -> None:
    """Start the FastAPI REST server."""
    import os
    if gateway:
        os.environ["LANKAMIND_GATEWAY"] = gateway
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
    )


# ── status ────────────────────────────────────────────────────────────────────

@cli.command("status")
@click.option("--gateway", default="tcp://localhost:5701", show_default=True,
              help="Gateway client port")
def status(gateway) -> None:
    """Show gateway and worker pool status."""
    import zmq, json

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 3_000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(gateway)
    try:
        sock.send_json({"type": "status"})
        resp = sock.recv_json()
    except zmq.Again:
        click.echo("Gateway not responding.")
        return
    finally:
        sock.close()
        ctx.term()

    workers = resp.get("workers", [])
    click.echo(f"\nGateway: {gateway}")
    click.echo(f"Workers: {len(workers)} registered")
    click.echo(f"Requests served: {resp.get('requests_total', '?')}\n")

    if workers:
        click.echo(f"{'ID':<14}  {'Shard':>5}  {'Host':<16}  {'Port':>5}  {'Latency':>10}  {'Healthy'}")
        click.echo("─" * 65)
        for w in sorted(workers, key=lambda x: x["shard_idx"]):
            click.echo(
                f"{w['worker_id']:<14}  {w['shard_idx']:>5}  {w['host']:<16}  "
                f"{w['port']:>5}  {w['latency_ms']:>8.1f} ms  {'✓' if w['is_healthy'] else '✗'}"
            )
    click.echo()


# ── balance ───────────────────────────────────────────────────────────────────

@cli.command("balance")
@click.argument("worker_id", required=False)
@click.option("--ledger", "ledger_path", default=None, help="Path to ledger.json")
def balance(worker_id, ledger_path) -> None:
    """Show reward ledger balance(s)."""
    from trust.ledger import Ledger, DEFAULT_LEDGER_PATH

    path = pathlib.Path(ledger_path) if ledger_path else DEFAULT_LEDGER_PATH
    ledger = Ledger(path=path)

    if worker_id:
        bal = ledger.get_balance(worker_id)
        click.echo(f"{worker_id}: {bal:.6f} LKM")
    else:
        all_b = ledger.get_all_balances()
        if not all_b:
            click.echo("No entries in ledger.")
            return
        for wid, bal in sorted(all_b.items()):
            click.echo(f"{wid}: {bal:.6f} LKM")
        click.echo(f"\nTotal tokens generated: {ledger.total_tokens_generated}")


# ── keys ──────────────────────────────────────────────────────────────────────

@cli.command("keys")
@click.option("--key-file", default=None, help="Path to node.key (default: ~/.lankamind/node.key)")
@click.option("--generate", "do_generate", is_flag=True, default=False,
              help="Generate a new keypair and save it")
def keys(key_file, do_generate) -> None:
    """Show or generate this node's Curve25519 keypair."""
    from network.keypair import get_or_create, generate as gen_kp, save as save_kp

    default_path = pathlib.Path.home() / ".lankamind" / "node.key"
    path = pathlib.Path(key_file) if key_file else default_path

    if do_generate:
        pub, sec = gen_kp()
        save_kp(path, pub, sec)
        click.echo(f"New keypair saved to: {path}")
    else:
        pub, _sec = get_or_create(path)

    click.echo(f"Public key : {pub}")
    click.echo(f"Key file   : {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cli()


if __name__ == "__main__":
    main()

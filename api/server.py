"""
api/server.py
-------------
FastAPI REST server for LankaMind.

Endpoints
---------
POST /v1/complete      — text completion via the distributed pipeline
GET  /v1/status        — gateway + worker status
GET  /v1/nodes         — list of registered worker nodes
GET  /v1/network-info  — LAN IP + mDNS name for mobile QR code
GET  /health           — simple liveness probe
GET  /                 — mobile web UI (served from api/static/index.html)

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Or via the CLI:
    lankamind api --port 8000
"""

from __future__ import annotations

import os
import pathlib
import socket
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Static file paths ─────────────────────────────────────────────────────────
_STATIC_DIR = pathlib.Path(__file__).parent / "static"

app = FastAPI(
    title="LankaMind API",
    description=(
        "Distributed LLM inference network for Sri Lanka. "
        "Run GPT-2 (and future models) split across multiple machines."
    ),
    version="0.5.0",
)

# Allow any origin (suitable for a public, open-source network)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (mobile web UI)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_START_TIME = time.time()


# ── Request / Response models ─────────────────────────────────────────────────

class CompleteRequest(BaseModel):
    prompt: str = Field(..., description="Text to continue", min_length=1)
    max_tokens: int = Field(60, ge=1, le=500, description="Maximum tokens to generate")
    model: str = Field("gpt2", description="Model name")
    workers: int = Field(3, ge=1, le=10, description="Number of pipeline workers")

    model_config = {"json_schema_extra": {
        "example": {
            "prompt": "The history of Sri Lanka spans",
            "max_tokens": 80,
            "model": "gpt2",
            "workers": 3,
        }
    }}


class CompleteResponse(BaseModel):
    prompt: str
    generated_text: str
    model: str
    tokens_generated: int
    elapsed_seconds: float


class NodeInfo(BaseModel):
    worker_id: str
    shard_idx: int
    num_shards: int
    model_name: str
    host: str
    port: int
    latency_ms: float
    is_healthy: bool


class StatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    active_workers: int
    healthy_workers: int
    requests_served: int
    gateway_address: Optional[str]


class NetworkInfoResponse(BaseModel):
    lan_ip: str
    port: int
    url: str
    mdns_url: str
    qr_hint: str


# ── Config (from env vars, with sensible defaults) ────────────────────────────

GATEWAY_ADDRESS: Optional[str] = os.environ.get("LANKAMIND_GATEWAY", None)
BASE_PORT: int   = int(os.environ.get("LANKAMIND_BASE_PORT",   "5500"))
RESULT_PORT: int = int(os.environ.get("LANKAMIND_RESULT_PORT", "5599"))
API_PORT: int    = int(os.environ.get("LANKAMIND_API_PORT",    "8000"))

_requests_served: int = 0


def _local_ip() -> str:
    """Return the machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ── Root — serve mobile web UI ────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the mobile web UI."""
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return HTMLResponse("<h1>LankaMind API</h1><p>See <a href='/docs'>/docs</a></p>")


@app.get("/manifest.json", include_in_schema=False)
async def manifest() -> FileResponse:
    """PWA manifest for mobile 'Add to Home Screen'."""
    path = _STATIC_DIR / "manifest.json"
    if path.exists():
        return FileResponse(str(path), media_type="application/manifest+json")
    raise HTTPException(404)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/complete", response_model=CompleteResponse)
async def complete(req: CompleteRequest) -> CompleteResponse:
    """Generate a text completion by sending the prompt through the worker pipeline."""
    global _requests_served

    from cli.client import run_client

    t0 = time.time()
    try:
        generated = run_client(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            model=req.model,
            workers=req.workers,
            base_port=BASE_PORT,
            result_port=RESULT_PORT,
            gateway_address=GATEWAY_ADDRESS,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc

    elapsed = time.time() - t0
    _requests_served += 1

    return CompleteResponse(
        prompt=req.prompt,
        generated_text=generated,
        model=req.model,
        tokens_generated=len(generated.split()),
        elapsed_seconds=round(elapsed, 3),
    )


@app.get("/v1/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Return gateway and worker pool status."""
    uptime  = time.time() - _START_TIME
    active  = 0
    healthy = 0

    if GATEWAY_ADDRESS:
        try:
            import zmq
            ctx  = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 2_000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(GATEWAY_ADDRESS)
            sock.send_json({"type": "status"})
            resp    = sock.recv_json()
            workers = resp.get("workers", [])
            active  = len(workers)
            healthy = sum(1 for w in workers if w.get("is_healthy"))
            sock.close()
            ctx.term()
        except Exception:
            pass

    return StatusResponse(
        status="ok",
        uptime_seconds=round(uptime, 1),
        active_workers=active,
        healthy_workers=healthy,
        requests_served=_requests_served,
        gateway_address=GATEWAY_ADDRESS,
    )


@app.get("/v1/nodes", response_model=List[NodeInfo])
async def nodes() -> List[NodeInfo]:
    """Return list of registered worker nodes from the gateway."""
    if not GATEWAY_ADDRESS:
        return []

    try:
        import zmq
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 2_000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(GATEWAY_ADDRESS)
        sock.send_json({"type": "status"})
        resp = sock.recv_json()
        sock.close()
        ctx.term()
        return [NodeInfo(**w) for w in resp.get("workers", [])]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Gateway unavailable: {exc}") from exc


@app.get("/v1/network-info", response_model=NetworkInfoResponse)
async def network_info() -> NetworkInfoResponse:
    """
    Return this machine's LAN IP and mDNS URL.

    Use this to generate a QR code so phones on the same Wi-Fi can connect
    by simply scanning it — no typing required.
    """
    ip   = _local_ip()
    port = API_PORT
    url  = f"http://{ip}:{port}"
    return NetworkInfoResponse(
        lan_ip=ip,
        port=port,
        url=url,
        mdns_url=f"http://lankamind.local:{port}",
        qr_hint=f"Scan to open LankaMind on your phone: {url}",
    )


@app.get("/health")
async def health() -> dict:
    """Simple liveness probe for Docker / Kubernetes health checks."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _START_TIME, 1)}

"""
api/server.py
-------------
FastAPI REST server for LankaMind.

Endpoints
---------
POST /v1/complete    — text completion via the distributed pipeline
GET  /v1/status      — gateway + worker status
GET  /v1/nodes       — list of registered worker nodes
GET  /health         — simple liveness probe

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Or via the CLI:
    lankamind api --port 8000
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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


# ── Config (from env vars, with sensible defaults) ────────────────────────────

GATEWAY_ADDRESS: Optional[str] = os.environ.get("LANKAMIND_GATEWAY", None)
BASE_PORT: int = int(os.environ.get("LANKAMIND_BASE_PORT", "5500"))
RESULT_PORT: int = int(os.environ.get("LANKAMIND_RESULT_PORT", "5599"))

_requests_served: int = 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/complete", response_model=CompleteResponse)
async def complete(req: CompleteRequest) -> CompleteResponse:
    """
    Generate a text completion by sending the prompt through the worker pipeline.
    """
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
    uptime = time.time() - _START_TIME
    active = 0
    healthy = 0

    if GATEWAY_ADDRESS:
        try:
            import zmq
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 2_000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(GATEWAY_ADDRESS)
            sock.send_json({"type": "status"})
            resp = sock.recv_json()
            sock.close()
            ctx.term()
            workers = resp.get("workers", [])
            active = len(workers)
            healthy = sum(1 for w in workers if w.get("is_healthy"))
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
        ctx = zmq.Context()
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


@app.get("/health")
async def health() -> dict:
    """Simple liveness probe for Docker / Kubernetes health checks."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _START_TIME, 1)}

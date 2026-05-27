# LankaMind — Phase Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | Core inference prototype — sharded GPT-2, ZMQ pipeline, CLI client | ✅ Done |
| **2** | Orchestration — node registry, scheduler, Prometheus metrics, 8-bit quant | 🔜 Next |
| **3** | Real P2P — hivemind/libp2p, multi-machine, health checks, NAT traversal | 🗓 Planned |
| **4** | Trust & incentives — reputation scoring, off-chain reward ledger, smart contracts | 🗓 Planned |
| **5** | Production API & UI — FastAPI access tiers, Next.js portal, Grafana dashboards | 🗓 Planned |

---

## Phase 2 Detail (next sprint)

**Goal:** move from "it works on one machine" to "it works on a LAN with
real load balancing and monitoring."

Deliverables:
- `orchestrator/registry.py` — in-memory node registry with TTL heartbeats
- `orchestrator/scheduler.py` — latency-aware chain builder
- Workers expose `/metrics` endpoint; Prometheus scrapes it
- Grafana dashboard: requests/s, p95 latency, active nodes
- `core/quantization.py` — 8-bit loading via bitsandbytes (GPU workers)
- KV-cache in `core/model_shard.py` (avoids re-computing past tokens)

**Done looks like:**
  ```
  # On machine A (GPU node):
  python scripts/launch_workers.py --shards 2 --model gpt2-xl

  # On machine B (CPU node):
  python scripts/launch_workers.py --shards 2 --model gpt2-xl --join A_IP

  # On any machine:
  python cli/client.py "Sri Lanka has" --max-tokens 100
  ```

---

## Phase 3 Detail

**Goal:** no central coordinator — nodes find each other on the internet.

- Replace ZMQ with libp2p streams (using the `py-libp2p` or `hivemind` library)
- DHT peer discovery (Kademlia-style): nodes announce themselves with TTL
- NAT hole-punching via relay nodes
- Multi-model support: workers advertise which models they hold

---

## Phase 4 Detail

**Goal:** node operators earn tokens for contributing compute.

- Spot-check verification: randomly re-run 1% of requests on a trusted node and
  compare outputs; penalise nodes that diverge
- Reputation score stored in a signed Merkle log (off-chain v1)
- EVM smart contract for settlement (Polygon or similar L2) — v2
- Dashboard showing top-earning nodes

---

## Phase 5 Detail

**Goal:** a polished product three types of users can actually use.

- FastAPI service with JWT auth and three access tiers
- `citizen`: 10 requests/min, free
- `private_sector`: 1000 requests/min, billed per 1k tokens
- `government`: dedicated pipeline with SLA, custom models
- Next.js chat portal (simple, mobile-friendly)
- Admin panel: live node map, request logs, billing

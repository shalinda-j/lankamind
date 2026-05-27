# 🌴 LankaMind

**A decentralized LLM inference network for Sri Lanka — and the world.**

Run large language models split across multiple machines, no central server required.  
Free, open-source, and built for the community.

[![CI](https://github.com/sachinthyamakulasooriya/lankamind/actions/workflows/ci.yml/badge.svg)](https://github.com/sachinthyamakulasooriya/lankamind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## What is LankaMind?

LankaMind lets you run GPT-2 (and future larger models) **split across several computers**,  
coordinated over the regular internet. Each participant contributes compute and earns  
LKM reward tokens in return.

```
You → Gateway → Worker 0 → Worker 1 → Worker 2 → You
               (layers 0-3) (4-7)      (8-11)
```

**Key properties:**
- ✅ **Decentralized** — no single server controls inference
- ✅ **Encrypted** — ZMQ CURVE (Curve25519) end-to-end encryption
- ✅ **Incentivized** — workers earn LKM tokens per generated token
- ✅ **Fault-tolerant** — gateway routes around failed workers
- ✅ **Easy setup** — one command to join the network

---

## Quick Install

```bash
git clone https://github.com/sachinthyamakulasooriya/lankamind.git
cd lankamind
pip install -e .
```

**Start workers + run a completion in 2 commands:**

```bash
# Terminal 1
lankamind node

# Terminal 2
lankamind complete "The history of Sri Lanka spans"
```

See [QUICKSTART.md](QUICKSTART.md) for the full 5-minute guide.

---

## Docker

```bash
docker-compose up --build
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Sri Lanka is a beautiful island that"}'
```

---

## Architecture

| Layer | Component | Description |
|-------|-----------|-------------|
| **Core** | `core/model_shard.py` | Splits GPT-2 transformer blocks across workers |
| **Transport** | `transport/zmq_transport.py` | ZMQ PUSH/PULL pipeline |
| **Encryption** | `transport/secure_transport.py` | ZMQ CURVE (Curve25519) |
| **Orchestration** | `orchestrator/gateway.py` | Worker registry + chain builder |
| **Health** | `orchestrator/health_checker.py` | Ping-based liveness monitoring |
| **P2P** | `network/bootstrap.py` | Peer discovery node |
| **Trust** | `trust/reputation.py` | EMA reputation scoring |
| **Rewards** | `trust/ledger.py` | Off-chain LKM token ledger |
| **API** | `api/server.py` | FastAPI REST interface |
| **CLI** | `cli/main.py` | Unified `lankamind` command |

---

## REST API

Start the server: `lankamind api`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/complete` | Text completion |
| `GET`  | `/v1/status`   | Gateway + worker status |
| `GET`  | `/v1/nodes`    | List active worker nodes |
| `GET`  | `/health`      | Liveness probe |

Docs at: http://localhost:8000/docs

---

## Running Tests

```bash
python -m pytest tests/unit/ -v
```

138 tests, all green.

---

## Project Roadmap

- [x] **Phase 1** — Pipeline parallelism (GPT-2 split across 3 workers)
- [x] **Phase 2** — Gateway orchestration + health checking
- [x] **Phase 3** — ZMQ CURVE encryption + peer discovery
- [x] **Phase 4** — EMA reputation + LKM reward ledger
- [x] **Phase 5** — REST API + Docker + unified CLI
- [ ] **Phase 6** — LLaMA-3 / Mistral support, on-chain settlement
- [ ] **Phase 7** — Web dashboard, mobile client

---

## License

MIT — free to use, modify, and distribute.  
See [LICENSE](LICENSE).

---

## Contributing

PRs welcome! Please run the test suite before submitting.

```bash
python -m pytest tests/unit/ -v
```

Join the discussion: [GitHub Issues](https://github.com/sachinthyamakulasooriya/lankamind/issues)

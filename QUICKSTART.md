# LankaMind — Quick Start Guide

Get up and running in **5 minutes** on a single machine.

---

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Git
- ~1 GB disk space (for GPT-2 model weights, downloaded once automatically)

---

## 1. Install

```bash
git clone https://github.com/sachinthyamakulasooriya/lankamind.git
cd lankamind
pip install -e .
```

> On Windows use `pip install -e .` inside a `cmd` or PowerShell window.

---

## 2. Start Workers (Terminal 1)

```bash
lankamind node --shards 3
```

Or with Python directly:
```bash
python scripts/launch_workers.py --shards 3
```

Wait until you see:
```
✓  All 3 workers ready in 12.3s
```

---

## 3. Run a Completion (Terminal 2)

```bash
lankamind complete "The history of Sri Lanka spans"
```

Or the classic Python way:
```bash
python cli/client.py "The history of Sri Lanka spans"
```

Expected output:
```
Prompt : The history of Sri Lanka spans
Output :  over 2,500 years. The island was first settled by the Sinhalese...
```

---

## 4. Interactive Chat (Terminal 2)

```bash
lankamind chat
```

Type a sentence start, press Enter, see the model continue it. `Ctrl-C` to quit.

---

## 5. With Gateway (auto-discovery)

For a multi-machine setup, start the gateway first:

**Terminal 1:**
```bash
lankamind gateway
```

**Terminal 2:**
```bash
lankamind node --shards 3 --gateway tcp://localhost:5700
```

**Terminal 3:**
```bash
lankamind complete "Sri Lanka" --gateway tcp://localhost:5701
```

---

## 6. REST API

```bash
lankamind api --port 8000
```

Then:
```bash
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Sri Lanka is a beautiful island that", "max_tokens": 60}'
```

API docs: http://localhost:8000/docs

---

## 7. Docker Compose (all-in-one)

```bash
docker-compose up --build
```

This starts: 1 gateway + 3 workers + 1 API server + 1 bootstrap node.

Then:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time in Colombo"}'
```

---

## 8. Run Tests

```bash
python -m pytest tests/unit/ -v
```

---

## Architecture Overview

```
Client / API
    │
    │ ZMQ PUSH (tokens)
    ▼
Worker 0  ──► Worker 1  ──► Worker 2
(layers 0-3)  (layers 4-7)  (layers 8-11)
    │
    │ ZMQ PULL (next token ID)
    ▼
Client / API

All workers send heartbeats to:
    Gateway (port 5700) ◄── Worker heartbeats (PULL)
    Gateway (port 5701) ──► Client discovery (REP)
```

---

## Useful Commands

| Command | Description |
|---------|-------------|
| `lankamind complete "text"` | One-shot completion |
| `lankamind chat` | Interactive REPL |
| `lankamind node` | Start workers |
| `lankamind gateway` | Start gateway |
| `lankamind api` | Start REST API |
| `lankamind status` | Show gateway status |
| `lankamind balance` | Show reward ledger |
| `lankamind keys` | Show/generate keypair |

---

## Troubleshooting

**Workers not starting?**
- Check `logs/worker_0.log` etc.
- Make sure ports 5500–5502, 5599 are free.

**Timeout after 60 s?**
- Are all 3 workers running? Run `lankamind node` first.
- First run downloads ~500 MB of model weights — be patient.

**"CURVE not available"?**
- Your libzmq was compiled without libsodium. Encryption falls back to plaintext automatically.
- To enable: `pip install pyzmq --force-reinstall` with a libsodium-enabled build.

---

## Adding More Models

Workers default to `gpt2`. To use a larger model:

```bash
lankamind node --model gpt2-medium --shards 4
lankamind complete "Sri Lanka" --model gpt2-medium --workers 4
```

Supported: any HuggingFace GPT-2 family model.
Phase 6 will add LLaMA-3 / Mistral support.

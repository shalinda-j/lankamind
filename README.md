# LankaMind — Decentralized LLM Network

> **Phase 1 — Core Inference Prototype**  
> A real language model split across multiple processes, communicating like
> separate machines, producing real text output end-to-end.

---

## What this is

LankaMind runs a large language model split across many ordinary devices.
Instead of one powerful server, every participant contributes a slice of the
model.  A query travels through the chain of devices; each one does a small
piece of the computation and passes the result to the next.

**Phase 1 simulates this on a single computer** using separate Python
processes — one per "shard" of the model.  This proves the pipeline concept
before we wire up real machines in Phase 2.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10 or newer** | Check: `python --version` |
| **~1.5 GB free RAM** | Each of the 3 workers loads its model slice |
| **~600 MB free disk** | GPT-2 weights download once, then cached |
| **Internet (first run only)** | Downloads GPT-2 from HuggingFace |
| **No GPU needed** | Everything runs on CPU for Phase 1 |
| **No paid service needed** | GPT-2 is free and public |

---

## Installation

```bash
# 1. Clone / enter the project
cd lankamind

# 2. Create and activate a virtual environment  (strongly recommended)
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate

# 3. Install the project and its dependencies
pip install -e ".[dev]"
```

> **What `pip install -e .` does:**  installs the project in "editable" mode,
> meaning Python can find the `core`, `transport`, and `cli` packages from
> anywhere on your machine without copying files.

---

## Running Phase 1

You need **two terminal windows** open at the same time.

### Terminal 1 — Start the workers

```bash
python scripts/launch_workers.py
```

This starts 3 worker processes.  Each one loads its slice of GPT-2.
You will see something like:

```
Starting 3 workers for model 'gpt2' …

  ✓  Worker 0  (shard 0/2)  PID 12345  port 5500  →  log: logs/worker_0.log
  ✓  Worker 1  (shard 1/2)  PID 12346  port 5501  →  log: logs/worker_1.log
  ✓  Worker 2  (shard 2/2)  PID 12347  port 5502  →  log: logs/worker_2.log

Waiting for model shards to load (timeout 180s) …
(Each worker downloads ~500 MB on the very first run — this is normal)

──────────────────────────────────────────────────────────
  ✓  All 3 workers ready in 28.4s
──────────────────────────────────────────────────────────

Run the client in a NEW terminal window:

    python cli/client.py "Sri Lanka is a beautiful island that"

Press Ctrl-C here to stop all workers.
```

**The first run takes longer** because GPT-2 is downloaded.  Subsequent
runs are fast (model is cached in `~/.cache/huggingface/`).

Options:
```bash
python scripts/launch_workers.py --shards 4    # use 4 workers instead of 3
python scripts/launch_workers.py --shards 2    # use 2 workers
```

### Terminal 2 — Send a prompt

```bash
python cli/client.py "Sri Lanka is a beautiful island that"
```

Expected output:
```
Loading tokeniser for 'gpt2' …

Prompt : Sri Lanka is a beautiful island that
Output :  has been the focus of a great deal of attention in recent years.
          The island is home to a number of...

✓  Generation complete (or EOS reached).
```

More examples:
```bash
# Shorter output
python cli/client.py "The ancient city of Anuradhapura" --max-tokens 30

# Longer story
python cli/client.py "Once upon a time, a fisherman in Colombo" --max-tokens 100
```

> **GPT-2 tip:** this model continues text, it doesn't answer questions.
> Start with a sentence fragment like `"Colombo is"` rather than a question
> like `"What is Colombo?"`.

### Stopping the workers

Press **Ctrl-C** in Terminal 1, or from any terminal:

```bash
python scripts/stop_workers.py
```

---

## Running the tests

```bash
# Fast unit tests only (no model download, no network, ~5 seconds)
pytest tests/unit/ -v

# Full integration test (spawns workers, downloads GPT-2 if not cached, ~2 min)
pytest tests/integration/ -v -s -m integration

# Everything
pytest -v
```

What each test suite covers:

| Suite | What it tests | Speed |
|-------|---------------|-------|
| `tests/unit/test_model_shard.py` | Layer splitting, tensor shapes, forward pass correctness | Fast |
| `tests/unit/test_zmq_transport.py` | Tensor serialise → send → deserialise round-trip | Fast |
| `tests/unit/test_pipeline.py` | Port numbers, addresses, topology config | Fast |
| `tests/integration/test_end_to_end.py` | Full subprocess pipeline with real GPT-2 | Slow |

To skip the slow integration test in CI:
```bash
SKIP_INTEGRATION=1 pytest tests/
```

---

## How the code is organised

```
lankamind/
├── core/                  ← The inference engine (Phase 1 ✅)
│   ├── model_shard.py     ← Splits GPT-2 layers; runs forward pass
│   ├── worker.py          ← One process per shard; ZMQ loop
│   ├── pipeline.py        ← Port/address config shared by worker + client
│   └── quantization.py    ← 8-bit quant stub (Phase 2)
│
├── transport/             ← How workers talk to each other
│   ├── zmq_transport.py   ← Tensor serialisation + ZMQ sockets (Phase 1 ✅)
│   └── libp2p_transport.py← Real P2P stub (Phase 3)
│
├── orchestrator/          ← Smart routing (Phase 2)
├── network/               ← P2P protocol (Phase 3)
├── trust/                 ← Incentives (Phase 4)
├── api/                   ← HTTP API (Phase 5)
├── frontend/              ← Web UI (Phase 5)
│
├── cli/client.py          ← The command you run to send a prompt
├── scripts/
│   ├── launch_workers.py  ← Start all workers
│   └── stop_workers.py    ← Stop all workers
│
├── tests/
│   ├── unit/              ← Fast, no network, no model download
│   └── integration/       ← Full end-to-end with real model
│
├── docs/
│   ├── architecture.md    ← How the layers fit together (with diagrams)
│   ├── phases.md          ← Roadmap for phases 2–5
│   └── phase1-notes.md    ← Engineering decisions made in Phase 1
│
└── logs/                  ← Worker output logs (worker_0.log, etc.)
```

---

## How the pipeline works (plain English)

1. You type a prompt: `"Colombo is the capital"`
2. The **client** turns it into numbers that the model understands
   (this is called *tokenisation*): `[Col, ombo, is, the, capital]` → `[8645, 78, 318, 262, 3139]`
3. Those numbers go over a network socket to **Worker 0**
4. Worker 0 turns the numbers into a table of floating-point values
   (the *embedding*), then runs its 4 transformer layers
5. The result — a big table of numbers called *hidden states* — travels to **Worker 1**
6. Worker 1 runs its 4 layers, passes to **Worker 2**
7. Worker 2 runs its 4 layers, then runs a final step that turns the hidden
   states into a probability score for every word in the vocabulary
8. The word with the highest score becomes the next output token
9. That token travels back to the **client**, which prints it and starts over
   with the extended sequence: `"Colombo is the capital of"`
10. Repeat until the model outputs a "stop" signal or we reach `--max-tokens`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'core'` | Run `pip install -e .` from the `lankamind/` directory |
| `Address already in use` (port 5500–5502) | Run `python scripts/stop_workers.py`, then try again |
| Client times out after 60 s | Workers may not have finished loading — check `logs/worker_*.log` |
| Workers crash immediately | Check Python version (`python --version` must be ≥ 3.10) |
| Very slow first run | Normal — GPT-2 is downloading (~500 MB). Wait for "All 3 workers ready". |
| Output looks like gibberish | GPT-2 is an old model. Try a clearer prompt starting mid-sentence. |

---

## Phase roadmap

| Phase | What gets added |
|-------|-----------------|
| ✅ 1 | Sharded GPT-2, ZMQ pipeline, CLI client |
| 🔜 2 | Node registry, smart scheduler, 8-bit GPU support, Prometheus metrics |
| 🗓 3 | libp2p P2P, multi-machine, peer discovery |
| 🗓 4 | Reputation scores, token reward ledger |
| 🗓 5 | FastAPI, Next.js portal, access tiers |

See `docs/phases.md` for detailed plans for each phase.

---

*Built as part of Project LankaMind.*

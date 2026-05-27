# LankaMind — Architecture Overview

## The Big Idea

Instead of running a large language model on one expensive server, LankaMind
splits the model across many ordinary devices.  Each device holds a few layers
of the model.  A user's query travels through a chain of devices — each one
does a little work and passes the result to the next — until a full answer is
produced.

This is called **pipeline parallelism**.  The same concept is used by:
- [Petals](https://github.com/bigscience-workshop/petals) (open-source)
- [BitTorrent](https://en.wikipedia.org/wiki/BitTorrent) (for files, not AI)

---

## System Layers

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1  Access & Application                           │
│           Chat UI  ·  Web portal  ·  Public API          │
│           Access tiers: Citizen / Private / Government   │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP / WebSocket
┌────────────────────────▼─────────────────────────────────┐
│  Layer 2  Orchestration & Routing                        │
│           Scheduler  ·  Load balancer  ·  Node registry  │
└────────────────────────┬─────────────────────────────────┘
                         │ ZMQ → libp2p (Phase 3)
┌────────────────────────▼─────────────────────────────────┐
│  Layer 3  Inference & Model Sharding       ← Phase 1     │
│           ModelShard  ·  Worker process  ·  KV-cache     │
│           8-bit quantisation (Phase 2)                   │
└────────────────────────┬─────────────────────────────────┘
                         │ activations (tensors)
┌────────────────────────▼─────────────────────────────────┐
│  Layer 4  Trust, Verification & Incentives               │
│           Proof-of-work  ·  Reputation  ·  Token ledger  │
└────────────────────────┬─────────────────────────────────┘
                         │ TCP / DHT
┌────────────────────────▼─────────────────────────────────┐
│  Layer 5  P2P Networking & Devices                       │
│           Peer discovery  ·  Health checks  ·  NAT relay │
└──────────────────────────────────────────────────────────┘
```

---

## Phase 1 — What is Actually Running

```
  ┌──────────┐   input_ids    ┌──────────┐  hidden_states  ┌──────────┐
  │ Worker 0 │ ─────────────► │ Worker 1 │ ───────────────► │ Worker 2 │
  │ Shard 0  │                │ Shard 1  │                  │ Shard 2  │
  │ layers   │                │ layers   │                  │ layers   │
  │  0 – 3   │                │  4 – 7   │                  │  8 – 11  │
  │ + embed  │                │          │                  │ + LM head│
  └──────────┘                └──────────┘                  └──────────┘
       ▲                                                          │
       │                                                          │ next_token_id
       │                                                          ▼
  ┌──────────┐  prompt text                              ┌──────────────┐
  │  Client  │ ────────────────────────────────────────► │   Results    │
  │(CLI app) │ ◄──────────────────────────────────────── │  port 5599   │
  └──────────┘  generated text                           └──────────────┘
```

### How one token is generated

1. Client tokenises the prompt: `"Hello"` → `[15496]`
2. Sends token IDs to Worker 0 via ZeroMQ PUSH socket (port 5500)
3. Worker 0 embeds tokens → runs transformer blocks 0–3 → sends hidden states to Worker 1
4. Worker 1 runs blocks 4–7 → sends hidden states to Worker 2
5. Worker 2 runs blocks 8–11 + LayerNorm + LM head → picks the highest-probability token → sends token ID back to client
6. Client appends the new token and repeats from step 2 until EOS or `--max-tokens`

---

## Key Design Decisions

| Decision | Chosen approach | Why |
|---|---|---|
| **Communication** | ZeroMQ PUSH/PULL | Zero config, fast, maps naturally to real network |
| **Serialisation** | `torch.save` / pickle | Handles all tensor dtypes; no schema needed |
| **Model** | GPT-2 (Phase 1) | Runs on CPU, no GPU required to prove the concept |
| **Sharding** | Equal layer ranges | Simple, predictable; load-aware splitting in Phase 2 |
| **Generation** | Greedy (argmax) | Simplest correct method; beam search / sampling in Phase 2 |
| **Auth / P2P** | None (Phase 1) | Keep the first prototype to its core goal |

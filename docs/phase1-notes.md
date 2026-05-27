# Phase 1 — Engineering Notes

## What was built

A proof-of-concept of pipeline-parallel LLM inference on a single machine,
simulating multiple physical devices as separate processes.

## Files created in Phase 1

| File | Purpose |
|------|---------|
| `core/model_shard.py` | Splits GPT-2 into N equal layer-blocks; runs forward pass |
| `core/worker.py` | Process that holds one shard; ZMQ listen → forward → send |
| `core/pipeline.py` | PipelineConfig: port numbers, addresses, topology |
| `transport/zmq_transport.py` | Serialize tensors; WorkerInputSocket / WorkerOutputSocket |
| `cli/client.py` | Tokenise → pipeline loop → print streaming output |
| `scripts/launch_workers.py` | Spawn N workers; wait for ready-file signals |
| `scripts/stop_workers.py` | Gracefully terminate workers by PID |

## Simplifications made (intentional for Phase 1)

1. **Greedy decoding only** — `argmax` at each step. No temperature, no top-k/top-p sampling.
   Phase 2 will add `transformers.GenerationConfig` support.

2. **No KV-cache** — each generation step re-processes the full sequence from scratch.
   This makes it O(n²) in time. KV-cache (Phase 2) makes it O(n).

3. **No GPU** — everything runs on CPU with float32 weights.
   Phase 2 adds `bitsandbytes` for 8-bit on GPU.

4. **Single client** — the result port (5599) is hardcoded to one client.
   Phase 2 adds a ROUTER socket with request-ID multiplexing for concurrent clients.

5. **No auth, no encryption** — plain TCP.
   Phase 3 adds libp2p noise protocol.

## Known limitations

- GPT-2 is a *completion* model, not a chat assistant. Prompts should be sentence
  starters, not questions.
- Generation quality is poor for long outputs because GPT-2 is a 2019 model.
  We use it purely to prove the pipeline concept — a LLaMA-3 shard can drop in
  at Phase 2.
- On Windows, Ctrl-C in the client terminal may not immediately terminate; use
  `python scripts/stop_workers.py` to clean up.

## Test coverage

| Test file | What it covers |
|-----------|----------------|
| `tests/unit/test_model_shard.py` | Layer splitting, shapes, forward pass |
| `tests/unit/test_zmq_transport.py` | Tensor serialisation round-trip |
| `tests/unit/test_pipeline.py` | Port arithmetic in PipelineConfig |
| `tests/integration/test_end_to_end.py` | Full subprocess pipeline (needs ~1.5 GB RAM) |

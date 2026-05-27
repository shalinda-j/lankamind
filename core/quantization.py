"""
core/quantization.py
--------------------
8-bit quantisation helpers using bitsandbytes.  (Phase 2 — stub only.)

Why quantisation?
  A full-precision (float32) GPT-2 weight matrix uses 4 bytes per number.
  8-bit quantisation keeps only 1 byte per number, cutting model size in
  half while losing only ~1% of accuracy.  This lets ordinary laptops host
  larger model shards.

Phase 2 plan:
  1. Add bitsandbytes to requirements.txt.
  2. Replace ModelShard.__init__() to accept a `load_in_8bit=True` flag.
  3. Call GPT2LMHeadModel.from_pretrained(..., load_in_8bit=True, device_map="auto").
  4. Verify output quality with a simple perplexity check.
"""

# TODO (Phase 2): implement quantise_shard() and dequantise_shard()

def quantise_shard(shard):  # noqa: ANN001
    raise NotImplementedError("8-bit quantisation is planned for Phase 2.")

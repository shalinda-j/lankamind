"""
core/model_shard.py
-------------------
ModelShard: a single node's slice of the GPT-2 transformer.

HOW SHARDING WORKS
==================
GPT-2 (small) has 12 identical "transformer blocks" stacked on top of each
other.  We split those 12 blocks as evenly as possible across N workers:

    Worker 0  —  embedding layer  +  blocks [0 … k)      → hidden states
    Worker 1  —  blocks [k … 2k)                          → hidden states
    …
    Worker N-1 —  blocks [… 12) + final LayerNorm + LM head → logits

Activations (tensors of shape [batch, seq_len, hidden_dim]) travel through
each worker in sequence, like water flowing through pipes.

DESIGN NOTES
============
• We load the FULL model once, copy the slice we need, then discard the rest.
  Peak RAM during loading is ~1× model size (≈ 500 MB for GPT-2 small), but
  steady-state per worker is only 1/N of that.

• forward() is decorated with @torch.no_grad() — we never train here, so we
  skip the gradient machinery for speed and lower memory.

• GPT2Block.forward() returns a *tuple* not a bare tensor; we always take [0].

Phase 2 additions (not here yet):
  - KV-cache (past_key_values) for faster autoregressive generation
  - 8-bit quantisation via bitsandbytes
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


# ── Helpers ───────────────────────────────────────────────────────────────────


def compute_layer_range(
    total_layers: int,
    shard_idx: int,
    num_shards: int,
) -> Tuple[int, int]:
    """
    Divide *total_layers* as evenly as possible across *num_shards*.
    Any remainder is absorbed by the last shard (it gets a few extra blocks).

    Returns (start, end) so the worker owns  blocks[start : end].

    Examples (12 layers):
        num_shards=3  → (0,4), (4,8), (8,12)
        num_shards=4  → (0,3), (3,6), (6,9), (9,12)
        num_shards=2  → (0,6), (6,12)
    """
    base = total_layers // num_shards
    start = shard_idx * base
    end = start + base if shard_idx < num_shards - 1 else total_layers
    return (start, end)


# ── Main class ────────────────────────────────────────────────────────────────


class ModelShard(nn.Module):
    """
    One worker's portion of a GPT-2 model.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID (e.g. "gpt2") or a local path.
        Ignored when *config* is supplied (used in unit tests).
    shard_idx : int
        Zero-based index of this shard (0 = first worker).
    num_shards : int
        Total number of workers in the pipeline.
    config : GPT2Config | None
        If provided, build the model from this config instead of downloading
        weights.  Used in tests to avoid a 500 MB download.
    """

    def __init__(
        self,
        model_name_or_path: str,
        shard_idx: int,
        num_shards: int,
        config: Optional[GPT2Config] = None,
    ) -> None:
        super().__init__()

        if num_shards < 1:
            raise ValueError("num_shards must be ≥ 1")
        if not (0 <= shard_idx < num_shards):
            raise ValueError(f"shard_idx {shard_idx} out of range for {num_shards} shards")

        self.shard_idx = shard_idx
        self.num_shards = num_shards
        self.is_first: bool = shard_idx == 0
        self.is_last: bool = shard_idx == num_shards - 1

        # ── Load the full model, extract our portion, discard the rest ────────
        if config is not None:
            full_model = GPT2LMHeadModel(config)
        else:
            full_model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

        transformer = full_model.transformer
        total_layers = len(transformer.h)

        self.layer_range: Tuple[int, int] = compute_layer_range(
            total_layers, shard_idx, num_shards
        )
        start, end = self.layer_range

        # ── Embedding (first shard only) ──────────────────────────────────────
        # wte: turns integer token IDs → dense vectors  [vocab → hidden_dim]
        # wpe: adds positional information               [max_pos → hidden_dim]
        if self.is_first:
            self.wte = transformer.wte
            self.wpe = transformer.wpe
            self.drop = transformer.drop

        # ── Transformer blocks (every shard) ──────────────────────────────────
        # nn.ModuleList so PyTorch tracks parameters correctly
        self.blocks = nn.ModuleList(transformer.h[start:end])

        # ── Final norm + vocab projection (last shard only) ───────────────────
        # ln_f: LayerNorm applied after all transformer blocks
        # lm_head: linear layer projecting hidden_dim → vocab_size
        if self.is_last:
            self.ln_f = transformer.ln_f
            self.lm_head = full_model.lm_head

        # Free the part of the model we are not using
        del full_model

        # ModelShard is inference-only: switch to eval mode so Dropout is disabled
        # and two calls with the same input always produce the same output.
        self.eval()

    # ── Forward pass ──────────────────────────────────────────────────────────

    @torch.no_grad()  # inference only — no gradients needed
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run this shard's portion of the forward pass.

        Arguments
        ---------
        input_ids    : LongTensor [batch, seq_len]          — shard 0 only
        hidden_states: FloatTensor [batch, seq_len, C]      — shards 1 … N-1

        Returns
        -------
        hidden_states: FloatTensor [batch, seq_len, C]      — non-last shards
        logits       : FloatTensor [batch, seq_len, vocab]  — last shard
        """

        # ── Step 1: Embedding (shard 0 only) ──────────────────────────────────
        if self.is_first:
            if input_ids is None:
                raise ValueError("Shard 0 expects input_ids, got None")

            _batch, seq_len = input_ids.shape
            # Create position indices [0, 1, 2, … seq_len-1] for the batch
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)  # shape [1, seq_len]

            tok_emb = self.wte(input_ids)     # [B, T, hidden_dim]
            pos_emb = self.wpe(position_ids)  # [1, T, hidden_dim]
            hidden_states = self.drop(tok_emb + pos_emb)

        # ── Step 2: Transformer blocks ────────────────────────────────────────
        if hidden_states is None:
            raise ValueError(f"Shard {self.shard_idx} received None hidden_states")

        for block in self.blocks:
            # transformers <=4.x  block() returned a tuple: (hidden_states, present, ...)
            # transformers  5.x   block() returns the tensor directly
            # We handle both so the code works regardless of installed version.
            block_out = block(hidden_states)
            hidden_states = block_out[0] if isinstance(block_out, (tuple, list)) else block_out

        # ── Step 3: Decode to logits (last shard only) ────────────────────────
        if self.is_last:
            hidden_states = self.ln_f(hidden_states)           # [B, T, C]
            logits = self.lm_head(hidden_states)               # [B, T, vocab]
            return logits

        return hidden_states

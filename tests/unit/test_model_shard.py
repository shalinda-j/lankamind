"""
tests/unit/test_model_shard.py
-------------------------------
Unit tests for ModelShard and compute_layer_range.

These tests use a *tiny* GPT-2 config (4 layers, hidden dim 128) that is
constructed in-memory — no internet download required.

Run:
    pytest tests/unit/test_model_shard.py -v
"""

import pytest
import torch
from transformers import GPT2Config

from core.model_shard import ModelShard, compute_layer_range


# ── Shared tiny model config ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_cfg() -> GPT2Config:
    """
    A GPT-2 config with 4 transformer layers and a hidden dimension of 128.
    Creates instantly without downloading any weights.
    """
    return GPT2Config(
        n_embd=128,
        n_layer=4,
        n_head=4,
        vocab_size=512,
        n_positions=64,
        n_ctx=64,
    )


# ── compute_layer_range ───────────────────────────────────────────────────────

class TestComputeLayerRange:

    def test_even_split_3_shards(self):
        # 12 layers / 3 shards = 4 each
        assert compute_layer_range(12, 0, 3) == (0, 4)
        assert compute_layer_range(12, 1, 3) == (4, 8)
        assert compute_layer_range(12, 2, 3) == (8, 12)

    def test_even_split_2_shards(self):
        assert compute_layer_range(12, 0, 2) == (0, 6)
        assert compute_layer_range(12, 1, 2) == (6, 12)

    def test_uneven_split_remainder_goes_to_last(self):
        # 4 layers / 3 shards: shards 0 and 1 get 1 layer each, shard 2 gets 2
        assert compute_layer_range(4, 0, 3) == (0, 1)
        assert compute_layer_range(4, 1, 3) == (1, 2)
        assert compute_layer_range(4, 2, 3) == (2, 4)  # picks up the remainder

    def test_single_shard_owns_all_layers(self):
        assert compute_layer_range(12, 0, 1) == (0, 12)

    def test_no_gaps_or_overlaps(self):
        total = 12
        n = 5
        ranges = [compute_layer_range(total, i, n) for i in range(n)]
        # Verify contiguous
        for i in range(1, n):
            assert ranges[i][0] == ranges[i - 1][1], "Gap between shards"
        # Verify full coverage
        assert ranges[0][0] == 0
        assert ranges[-1][1] == total


# ── ModelShard structure ──────────────────────────────────────────────────────

class TestModelShardStructure:

    def test_first_shard_has_embedding_layers(self, tiny_cfg):
        shard = ModelShard("tiny", 0, 2, config=tiny_cfg)
        assert shard.is_first
        assert hasattr(shard, "wte"), "First shard must have token embedding (wte)"
        assert hasattr(shard, "wpe"), "First shard must have position embedding (wpe)"

    def test_last_shard_has_lm_head(self, tiny_cfg):
        shard = ModelShard("tiny", 1, 2, config=tiny_cfg)
        assert shard.is_last
        assert hasattr(shard, "lm_head"), "Last shard must have lm_head"
        assert hasattr(shard, "ln_f"),    "Last shard must have final LayerNorm"

    def test_middle_shard_no_embedding_no_head(self, tiny_cfg):
        # 4 layers / 4 shards → shard 1 is a middle shard
        shard = ModelShard("tiny", 1, 4, config=tiny_cfg)
        assert not shard.is_first
        assert not shard.is_last
        assert not hasattr(shard, "wte"),     "Middle shard must NOT have wte"
        assert not hasattr(shard, "lm_head"), "Middle shard must NOT have lm_head"

    def test_correct_number_of_blocks(self, tiny_cfg):
        # tiny_cfg has 4 layers; 2 shards → 2 blocks each
        shard0 = ModelShard("tiny", 0, 2, config=tiny_cfg)
        shard1 = ModelShard("tiny", 1, 2, config=tiny_cfg)
        assert len(shard0.blocks) == 2
        assert len(shard1.blocks) == 2

    def test_single_shard_has_everything(self, tiny_cfg):
        shard = ModelShard("tiny", 0, 1, config=tiny_cfg)
        assert shard.is_first
        assert shard.is_last
        assert len(shard.blocks) == 4  # all 4 layers


# ── ModelShard forward pass shapes ───────────────────────────────────────────

class TestModelShardForward:

    def test_first_shard_output_shape(self, tiny_cfg):
        """First shard embeds tokens → should output [batch, seq, hidden]."""
        shard = ModelShard("tiny", 0, 2, config=tiny_cfg)
        ids = torch.randint(0, 512, (1, 10))   # batch=1, seq_len=10
        out = shard(input_ids=ids)
        assert out.shape == (1, 10, 128), f"Expected (1,10,128), got {out.shape}"

    def test_last_shard_output_shape(self, tiny_cfg):
        """Last shard receives hidden states → should output logits [batch, seq, vocab]."""
        shard0 = ModelShard("tiny", 0, 2, config=tiny_cfg)
        shard1 = ModelShard("tiny", 1, 2, config=tiny_cfg)

        ids = torch.randint(0, 512, (1, 10))
        hidden = shard0(input_ids=ids)          # [1, 10, 128]
        logits = shard1(hidden_states=hidden)   # [1, 10, 512]

        assert logits.shape == (1, 10, 512), f"Expected (1,10,512), got {logits.shape}"

    def test_chain_of_3_shards(self, tiny_cfg):
        """Full 3-shard chain on the 4-layer tiny model."""
        shards = [ModelShard("tiny", i, 3, config=tiny_cfg) for i in range(3)]
        ids = torch.randint(0, 512, (1, 8))

        x = shards[0](input_ids=ids)
        for shard in shards[1:]:
            x = shard(hidden_states=x)

        # After the last shard we have logits
        assert x.shape[-1] == 512, "Logits should have vocab_size=512 as last dim"

    def test_first_shard_raises_without_input_ids(self, tiny_cfg):
        shard = ModelShard("tiny", 0, 2, config=tiny_cfg)
        with pytest.raises(ValueError, match="input_ids"):
            shard(hidden_states=torch.randn(1, 5, 128))

    def test_output_is_deterministic(self, tiny_cfg):
        """Two forward passes with the same input must give the same output."""
        shard = ModelShard("tiny", 0, 2, config=tiny_cfg)
        ids = torch.randint(0, 512, (1, 6))
        out1 = shard(input_ids=ids)
        out2 = shard(input_ids=ids)
        assert torch.allclose(out1, out2)

    def test_variable_sequence_length(self, tiny_cfg):
        """Shard should handle any sequence length up to n_positions."""
        shard = ModelShard("tiny", 0, 2, config=tiny_cfg)
        for seq_len in [1, 5, 32, 63]:
            ids = torch.randint(0, 512, (1, seq_len))
            out = shard(input_ids=ids)
            assert out.shape == (1, seq_len, 128)

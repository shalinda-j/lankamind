"""
trust/verifier.py
-----------------
Spot-check verifier for LankaMind worker outputs.

With probability SPOT_CHECK_RATE (default 5%), the verifier re-runs a
request through a trusted reference shard and compares the result.
A mismatch triggers a reputation penalty.

Design
------
• The verifier is a *probabilistic* check, not a full consensus.
  It's cheap (5% overhead) and catches accidental or malicious errors.
• Reference computation uses the same ModelShard on the gateway host;
  for Phase 4 this is always a local re-run on shard 0 of a known-good
  single-machine config.
• Future versions (Phase 6) will use cryptographic commitments.

Usage
-----
    verifier = SpotCheckVerifier(tracker, ledger, spot_rate=0.05)
    verifier.maybe_verify(
        worker_id="w1",
        input_ids=tensor,
        expected_next_token=42,
        model_name="gpt2",
    )
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import torch

from trust.reputation import ReputationTracker
from trust.ledger import Ledger

log = logging.getLogger(__name__)

SPOT_CHECK_RATE: float = 0.05       # 5% of requests are spot-checked
PENALTY_AMOUNT: float  = 0.01       # LKM deducted for a bad result


class SpotCheckVerifier:
    """
    Probabilistic spot-check verifier.

    Parameters
    ----------
    tracker   : ReputationTracker to update on verification outcome.
    ledger    : Ledger to apply penalties / rewards.
    spot_rate : Fraction of requests to re-verify (default 5%).
    rng_seed  : Optional seed for reproducible tests.
    """

    def __init__(
        self,
        tracker: ReputationTracker,
        ledger: Ledger,
        spot_rate: float = SPOT_CHECK_RATE,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.tracker   = tracker
        self.ledger    = ledger
        self.spot_rate = spot_rate
        self._rng = random.Random(rng_seed)

    def should_check(self) -> bool:
        """Return True with probability spot_rate."""
        return self._rng.random() < self.spot_rate

    def verify(
        self,
        worker_id: str,
        input_ids: torch.Tensor,
        expected_next_token: int,
        model_name: str = "gpt2",
        num_shards: int = 1,
    ) -> bool:
        """
        Re-run *input_ids* through a local single-shard model and compare
        the predicted next token to *expected_next_token*.

        Returns True if the result matches (honest worker), False otherwise.
        Updates reputation and (on failure) applies a ledger penalty.
        """
        try:
            from core.model_shard import ModelShard
            # Always use a single shard for reference (shard 0 of 1)
            ref_shard = ModelShard(model_name, shard_idx=0, num_shards=1)
            with torch.no_grad():
                logits = ref_shard(input_ids=input_ids)
            ref_token = int(logits[0, -1, :].argmax().item())
        except Exception as exc:
            log.warning("Verifier error for worker %s: %s", worker_id, exc)
            return True   # can't verify — give benefit of the doubt

        match = (ref_token == expected_next_token)

        if match:
            self.tracker.record_good(worker_id)
            log.debug("Spot-check PASS for worker %s (token %d)", worker_id, expected_next_token)
        else:
            self.tracker.record_bad(worker_id)
            self.ledger.debit(worker_id, PENALTY_AMOUNT)
            log.warning(
                "Spot-check FAIL for worker %s: expected %d, got %d",
                worker_id, expected_next_token, ref_token,
            )

        return match

    def maybe_verify(
        self,
        worker_id: str,
        input_ids: torch.Tensor,
        expected_next_token: int,
        model_name: str = "gpt2",
        num_shards: int = 1,
    ) -> Optional[bool]:
        """
        Verify with probability spot_rate.

        Returns True/False if a check was run, None if skipped.
        """
        if not self.should_check():
            # Record a good outcome optimistically (no check done)
            self.tracker.record_good(worker_id)
            return None
        return self.verify(worker_id, input_ids, expected_next_token, model_name, num_shards)

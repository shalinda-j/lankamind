"""
tests/unit/test_ledger.py
--------------------------
Unit tests for trust.ledger.Ledger.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from trust.ledger import REWARD_PER_TOKEN, Ledger


class TestLedger:
    def _make(self, tmp_path: pathlib.Path) -> Ledger:
        return Ledger(path=tmp_path / "ledger.json")

    def test_initial_balance_zero(self, tmp_path):
        ledger = self._make(tmp_path)
        assert ledger.get_balance("w1") == pytest.approx(0.0)

    def test_reward_single_token(self, tmp_path):
        ledger = self._make(tmp_path)
        balance = ledger.reward("w1", tokens=1)
        assert balance == pytest.approx(REWARD_PER_TOKEN)

    def test_reward_multiple_tokens(self, tmp_path):
        ledger = self._make(tmp_path)
        balance = ledger.reward("w1", tokens=10)
        assert balance == pytest.approx(10 * REWARD_PER_TOKEN)

    def test_reward_accumulates(self, tmp_path):
        ledger = self._make(tmp_path)
        ledger.reward("w1", tokens=5)
        balance = ledger.reward("w1", tokens=5)
        assert balance == pytest.approx(10 * REWARD_PER_TOKEN)

    def test_multiple_workers_independent(self, tmp_path):
        ledger = self._make(tmp_path)
        ledger.reward("w1", tokens=10)
        ledger.reward("w2", tokens=3)
        assert ledger.get_balance("w1") == pytest.approx(10 * REWARD_PER_TOKEN)
        assert ledger.get_balance("w2") == pytest.approx(3 * REWARD_PER_TOKEN)

    def test_debit_reduces_balance(self, tmp_path):
        ledger = self._make(tmp_path)
        ledger.reward("w1", tokens=100)
        balance_after = ledger.debit("w1", 0.05)
        assert balance_after < 100 * REWARD_PER_TOKEN

    def test_debit_clamped_at_zero(self, tmp_path):
        ledger = self._make(tmp_path)
        # Worker has 0 balance; debiting should clamp to 0, not go negative
        balance = ledger.debit("w1", 999.0)
        assert balance == pytest.approx(0.0)

    def test_total_tokens_tracked(self, tmp_path):
        ledger = self._make(tmp_path)
        ledger.reward("w1", tokens=7)
        ledger.reward("w2", tokens=3)
        assert ledger.total_tokens_generated == 10

    def test_get_all_balances(self, tmp_path):
        ledger = self._make(tmp_path)
        ledger.reward("w1", tokens=1)
        ledger.reward("w2", tokens=2)
        all_b = ledger.get_all_balances()
        assert set(all_b.keys()) == {"w1", "w2"}

    def test_save_creates_file(self, tmp_path):
        ledger = self._make(tmp_path)
        ledger.reward("w1", tokens=5)
        ledger.save()
        assert (tmp_path / "ledger.json").exists()

    def test_save_and_reload(self, tmp_path):
        ledger = self._make(tmp_path)
        ledger.reward("w1", tokens=42)
        ledger.save()

        ledger2 = self._make(tmp_path)  # loads on __init__
        assert ledger2.get_balance("w1") == pytest.approx(42 * REWARD_PER_TOKEN)
        assert ledger2.total_tokens_generated == 42

    def test_persist_json_format(self, tmp_path):
        ledger = self._make(tmp_path)
        ledger.reward("w1", tokens=1)
        ledger.save()
        data = json.loads((tmp_path / "ledger.json").read_text())
        assert "version" in data
        assert "balances" in data
        assert "total_tokens_generated" in data
        assert "w1" in data["balances"]

    def test_load_corrupt_file_no_crash(self, tmp_path):
        path = tmp_path / "ledger.json"
        path.write_text("not json at all")
        ledger = Ledger(path=path)   # should not raise
        assert ledger.get_balance("w1") == pytest.approx(0.0)

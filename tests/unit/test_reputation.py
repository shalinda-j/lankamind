"""
tests/unit/test_reputation.py
------------------------------
Unit tests for trust.reputation.ReputationTracker and ReputationEntry.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from trust.reputation import (
    ALPHA,
    INITIAL_SCORE,
    REPUTATION_THRESHOLD,
    ReputationEntry,
    ReputationTracker,
)


# ── ReputationEntry ───────────────────────────────────────────────────────────

class TestReputationEntry:
    def test_initial_score(self):
        e = ReputationEntry(worker_id="w1")
        assert e.score == pytest.approx(INITIAL_SCORE)

    def test_good_outcome_increases_score(self):
        e = ReputationEntry(worker_id="w1", score=0.5)
        new = e.update(1.0)
        assert new > 0.5

    def test_bad_outcome_decreases_score(self):
        e = ReputationEntry(worker_id="w1", score=0.8)
        new = e.update(0.0)
        assert new < 0.8

    def test_ema_formula(self):
        e = ReputationEntry(worker_id="w1", score=0.8)
        expected = ALPHA * 1.0 + (1 - ALPHA) * 0.8
        assert e.update(1.0) == pytest.approx(expected)

    def test_score_clamped_at_max(self):
        e = ReputationEntry(worker_id="w1", score=0.99)
        for _ in range(100):
            e.update(1.0)
        assert e.score <= 1.0

    def test_score_clamped_at_min(self):
        e = ReputationEntry(worker_id="w1", score=0.01)
        for _ in range(100):
            e.update(0.0)
        assert e.score >= 0.0

    def test_outcome_clamped_to_0_1(self):
        e = ReputationEntry(worker_id="w1", score=0.5)
        new = e.update(5.0)  # outcome > 1 should be clamped to 1
        assert new <= 1.0

    def test_total_updates_incremented(self):
        e = ReputationEntry(worker_id="w1")
        e.update(1.0)
        e.update(0.0)
        assert e.total_updates == 2

    def test_good_bad_outcome_counters(self):
        e = ReputationEntry(worker_id="w1")
        e.update(1.0)
        e.update(1.0)
        e.update(0.0)
        assert e.good_outcomes == 2
        assert e.bad_outcomes == 1

    def test_is_trusted_above_threshold(self):
        e = ReputationEntry(worker_id="w1", score=0.8)
        assert e.is_trusted

    def test_is_not_trusted_below_threshold(self):
        e = ReputationEntry(worker_id="w1", score=0.1)
        assert not e.is_trusted

    def test_to_dict_roundtrip(self):
        e = ReputationEntry(worker_id="w1", score=0.75, total_updates=5)
        d = e.to_dict()
        e2 = ReputationEntry.from_dict(d)
        assert e2.worker_id == "w1"
        assert e2.score == pytest.approx(0.75)
        assert e2.total_updates == 5


# ── ReputationTracker ─────────────────────────────────────────────────────────

class TestReputationTracker:
    def test_unknown_worker_returns_initial_score(self):
        t = ReputationTracker()
        assert t.get_score("unknown") == pytest.approx(INITIAL_SCORE)

    def test_record_good_increases_score(self):
        t = ReputationTracker()
        t._update("w1", 0.5)  # set a low baseline
        t.record_good("w1")
        assert t.get_score("w1") > 0.5

    def test_record_bad_decreases_score(self):
        t = ReputationTracker()
        initial = t.get_score("w1")  # creates entry on first access? No — just returns constant
        t._update("w1", 0.8)         # set a high baseline
        score_before = t.get_score("w1")
        t.record_bad("w1")
        assert t.get_score("w1") < score_before

    def test_record_outcome_fractional(self):
        t = ReputationTracker()
        t._update("w1", 0.5)
        score = t.record_outcome("w1", 0.6)
        assert 0.0 <= score <= 1.0

    def test_get_trusted_workers_filters_low_score(self):
        t = ReputationTracker()
        # Drive w1 down below threshold
        for _ in range(30):
            t.record_bad("w1")
        # w2 stays at default (trusted)
        t.record_good("w2")
        trusted = t.get_trusted_workers()
        assert "w1" not in trusted
        assert "w2" in trusted

    def test_snapshot_returns_list(self):
        t = ReputationTracker()
        t.record_good("w1")
        snap = t.snapshot()
        assert isinstance(snap, list)
        assert len(snap) == 1
        assert snap[0]["worker_id"] == "w1"

    def test_save_and_load(self, tmp_path: pathlib.Path):
        path = tmp_path / "rep.json"
        t = ReputationTracker()
        t.record_good("w1")
        t.record_bad("w2")
        t.save(path)

        t2 = ReputationTracker()
        t2.load(path)
        assert t2.get_score("w1") == pytest.approx(t.get_score("w1"))
        assert t2.get_score("w2") == pytest.approx(t.get_score("w2"))

    def test_load_nonexistent_file_no_error(self, tmp_path: pathlib.Path):
        t = ReputationTracker()
        t.load(tmp_path / "does_not_exist.json")  # should not raise

    def test_multiple_workers_independent(self):
        t = ReputationTracker()
        for _ in range(20):
            t.record_bad("bad_worker")
        for _ in range(20):
            t.record_good("good_worker")
        assert t.get_score("bad_worker") < t.get_score("good_worker")

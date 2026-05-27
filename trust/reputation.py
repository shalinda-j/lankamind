"""
trust/reputation.py
--------------------
Exponential-Moving-Average (EMA) reputation tracker for LankaMind nodes.

Each worker node starts with an initial reputation of 0.8 (out of 1.0) and
its score is updated every time it produces a verifiable result:

    score_new = alpha * outcome + (1 - alpha) * score_old

where:
  outcome = 1.0 for a correct/verified result
  outcome = 0.0 for a bad/failed/timeout result
  alpha   = 0.1  (default)

Constants
---------
INITIAL_SCORE = 0.8
ALPHA         = 0.1
MIN_SCORE     = 0.0
MAX_SCORE     = 1.0
"""

from __future__ import annotations

import json
import pathlib
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional


INITIAL_SCORE: float = 0.8
ALPHA: float = 0.1
MIN_SCORE: float = 0.0
MAX_SCORE: float = 1.0
REPUTATION_THRESHOLD: float = 0.3


@dataclass
class ReputationEntry:
    worker_id: str
    score: float = INITIAL_SCORE
    total_updates: int = 0
    good_outcomes: int = 0
    bad_outcomes: int = 0

    def update(self, outcome: float, alpha: float = ALPHA) -> float:
        outcome = max(0.0, min(1.0, outcome))
        self.score = alpha * outcome + (1.0 - alpha) * self.score
        self.score = max(MIN_SCORE, min(MAX_SCORE, self.score))
        self.total_updates += 1
        if outcome >= 0.5:
            self.good_outcomes += 1
        else:
            self.bad_outcomes += 1
        return self.score

    @property
    def is_trusted(self) -> bool:
        return self.score >= REPUTATION_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "score": self.score,
            "total_updates": self.total_updates,
            "good_outcomes": self.good_outcomes,
            "bad_outcomes": self.bad_outcomes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReputationEntry":
        e = cls(worker_id=d["worker_id"], score=d["score"])
        e.total_updates = d.get("total_updates", 0)
        e.good_outcomes = d.get("good_outcomes", 0)
        e.bad_outcomes = d.get("bad_outcomes", 0)
        return e


class ReputationTracker:
    """Thread-safe EMA reputation store for all known worker nodes."""

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = alpha
        self._lock = threading.RLock()
        self._entries: Dict[str, ReputationEntry] = {}

    def get_score(self, worker_id: str) -> float:
        with self._lock:
            entry = self._entries.get(worker_id)
            return entry.score if entry else INITIAL_SCORE

    def get_entry(self, worker_id: str) -> Optional[ReputationEntry]:
        with self._lock:
            return self._entries.get(worker_id)

    def get_trusted_workers(self, threshold: float = REPUTATION_THRESHOLD) -> List[str]:
        with self._lock:
            return [wid for wid, e in self._entries.items() if e.score >= threshold]

    def snapshot(self) -> List[dict]:
        with self._lock:
            return [e.to_dict() for e in self._entries.values()]

    def record_good(self, worker_id: str) -> float:
        return self._update(worker_id, 1.0)

    def record_bad(self, worker_id: str) -> float:
        return self._update(worker_id, 0.0)

    def record_outcome(self, worker_id: str, outcome: float) -> float:
        return self._update(worker_id, outcome)

    def _update(self, worker_id: str, outcome: float) -> float:
        with self._lock:
            if worker_id not in self._entries:
                self._entries[worker_id] = ReputationEntry(worker_id=worker_id)
            return self._entries[worker_id].update(outcome, self.alpha)

    def save(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [e.to_dict() for e in self._entries.values()]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            for d in data:
                self._entries[d["worker_id"]] = ReputationEntry.from_dict(d)

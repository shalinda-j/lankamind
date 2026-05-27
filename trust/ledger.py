"""
trust/ledger.py
---------------
Off-chain JSON reward ledger for LankaMind.

Every worker earns LKM tokens for each token they help generate.
Balances are stored in a local JSON file at ~/.lankamind/ledger.json.

This is intentionally NOT a blockchain ledger — private user data stays local.
A future on-chain settlement layer can batch-settle balances periodically
(Phase 6+), but the day-to-day accounting lives here.

Constants
---------
REWARD_PER_TOKEN = 0.001 LKM per generated token
DEFAULT_LEDGER_PATH = ~/.lankamind/ledger.json

Format (ledger.json)
--------------------
{
  "version": 1,
  "balances": {
    "<worker_id>": <float LKM balance>,
    ...
  },
  "total_tokens_generated": <int>
}
"""

from __future__ import annotations

import json
import pathlib
import threading
from typing import Dict

REWARD_PER_TOKEN: float = 0.001  # LKM per token generated
DEFAULT_LEDGER_PATH: pathlib.Path = pathlib.Path.home() / ".lankamind" / "ledger.json"
LEDGER_VERSION: int = 1


class Ledger:
    """
    Thread-safe off-chain reward ledger.

    Usage
    -----
        ledger = Ledger()
        ledger.reward(worker_id="w1", tokens=10)
        balance = ledger.get_balance("w1")
        ledger.save()
    """

    def __init__(
        self,
        path: pathlib.Path = DEFAULT_LEDGER_PATH,
        reward_per_token: float = REWARD_PER_TOKEN,
    ) -> None:
        self.path = pathlib.Path(path)
        self.reward_per_token = reward_per_token
        self._lock = threading.RLock()
        self._balances: Dict[str, float] = {}
        self._total_tokens: int = 0
        self._load_if_exists()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_balance(self, worker_id: str) -> float:
        """Return balance in LKM (0.0 if unknown)."""
        with self._lock:
            return self._balances.get(worker_id, 0.0)

    def get_all_balances(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._balances)

    @property
    def total_tokens_generated(self) -> int:
        with self._lock:
            return self._total_tokens

    # ── Write ─────────────────────────────────────────────────────────────────

    def reward(self, worker_id: str, tokens: int = 1) -> float:
        """
        Credit *worker_id* for *tokens* generated tokens.

        Returns
        -------
        float — the new balance after crediting.
        """
        amount = tokens * self.reward_per_token
        with self._lock:
            self._balances[worker_id] = self._balances.get(worker_id, 0.0) + amount
            self._total_tokens += tokens
            return self._balances[worker_id]

    def debit(self, worker_id: str, amount: float) -> float:
        """
        Debit *amount* LKM from *worker_id* (e.g. for a penalty).
        Balance is clamped at 0.0.

        Returns
        -------
        float — the new balance after debiting.
        """
        with self._lock:
            current = self._balances.get(worker_id, 0.0)
            self._balances[worker_id] = max(0.0, current - amount)
            return self._balances[worker_id]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the ledger to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "version": LEDGER_VERSION,
                "balances": dict(self._balances),
                "total_tokens_generated": self._total_tokens,
            }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_if_exists(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            with self._lock:
                self._balances = data.get("balances", {})
                self._total_tokens = data.get("total_tokens_generated", 0)
        except Exception:
            pass  # corrupt file — start fresh

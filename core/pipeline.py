"""
core/pipeline.py
----------------
PipelineConfig: a single source of truth for how workers are wired together.

Both the launch script and the CLI client import this so port numbers are
never duplicated or mis-matched by hand.

Phase 2 will extend this class to hold the full node registry pulled from
the orchestrator, not a static list of local ports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """
    Describes the layout of a local worker pipeline.

    Attributes
    ----------
    model_name  : HuggingFace model ID or local path.
    num_shards  : How many worker processes to use.
    base_port   : First worker listens on this port; worker i on base_port+i.
    result_port : Port on which the CLI client listens for final results.
    host        : Hostname/IP for all workers (localhost for Phase 1).
    """

    model_name: str
    num_shards: int
    base_port: int = 5500
    result_port: int = 5599
    host: str = "localhost"

    # ── Derived helpers ───────────────────────────────────────────────────────

    def worker_input_port(self, shard_idx: int) -> int:
        """TCP port that worker *shard_idx* binds its PULL socket on."""
        return self.base_port + shard_idx

    def worker_output_address(self, shard_idx: int) -> str:
        """
        The address a worker's PUSH socket connects to.

        For non-last workers this is the *next* worker's input port.
        For the last worker this is the client's result port.
        Returns empty string only when num_shards == 1 and we'd normally
        connect back to ourselves (degenerate single-shard case).
        """
        if shard_idx < self.num_shards - 1:
            return f"tcp://{self.host}:{self.base_port + shard_idx + 1}"
        # Last worker sends results to the client's receive port
        return f"tcp://{self.host}:{self.result_port}"

    def result_address(self) -> str:
        """Address the CLI client binds its PULL socket on."""
        return f"tcp://*:{self.result_port}"

    def first_worker_address(self) -> str:
        """Address the CLI client connects its PUSH socket to."""
        return f"tcp://{self.host}:{self.base_port}"

    def all_worker_ports(self) -> list[int]:
        """List of all worker input ports in chain order."""
        return [self.base_port + i for i in range(self.num_shards)]

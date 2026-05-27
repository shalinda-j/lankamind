"""
tests/unit/test_pipeline.py
----------------------------
Unit tests for PipelineConfig in core/pipeline.py.

Run:
    pytest tests/unit/test_pipeline.py -v
"""

import pytest
from core.pipeline import PipelineConfig


class TestPipelineConfig:

    def test_worker_input_ports(self):
        cfg = PipelineConfig("gpt2", num_shards=3, base_port=5500)
        assert cfg.worker_input_port(0) == 5500
        assert cfg.worker_input_port(1) == 5501
        assert cfg.worker_input_port(2) == 5502

    def test_output_address_middle_workers(self):
        cfg = PipelineConfig("gpt2", num_shards=3)
        # Worker 0 → Worker 1
        assert cfg.worker_output_address(0) == "tcp://localhost:5501"
        # Worker 1 → Worker 2
        assert cfg.worker_output_address(1) == "tcp://localhost:5502"

    def test_output_address_last_worker_goes_to_result_port(self):
        cfg = PipelineConfig("gpt2", num_shards=3)
        # Last worker sends to the result port
        assert cfg.worker_output_address(2) == "tcp://localhost:5599"

    def test_result_address(self):
        cfg = PipelineConfig("gpt2", num_shards=3)
        assert cfg.result_address() == "tcp://*:5599"

    def test_first_worker_address(self):
        cfg = PipelineConfig("gpt2", num_shards=3)
        assert cfg.first_worker_address() == "tcp://localhost:5500"

    def test_all_worker_ports(self):
        cfg = PipelineConfig("gpt2", num_shards=4)
        assert cfg.all_worker_ports() == [5500, 5501, 5502, 5503]

    def test_custom_ports(self):
        cfg = PipelineConfig("gpt2", num_shards=2, base_port=6000, result_port=6099)
        assert cfg.worker_input_port(0) == 6000
        assert cfg.worker_input_port(1) == 6001
        assert cfg.worker_output_address(1) == "tcp://localhost:6099"
        assert cfg.result_address() == "tcp://*:6099"

    def test_single_shard_output_goes_to_result(self):
        cfg = PipelineConfig("gpt2", num_shards=1)
        # With a single shard: it is both first and last
        # output_address should be the result port
        assert cfg.worker_output_address(0) == "tcp://localhost:5599"

    def test_custom_host(self):
        cfg = PipelineConfig("gpt2", num_shards=3, host="192.168.1.10")
        assert cfg.worker_output_address(0) == "tcp://192.168.1.10:5501"
        assert cfg.first_worker_address() == "tcp://192.168.1.10:5500"

    def test_no_port_collision(self):
        """Worker ports and result port must not overlap."""
        cfg = PipelineConfig("gpt2", num_shards=3)
        worker_ports = set(cfg.all_worker_ports())
        assert cfg.result_port not in worker_ports, \
            "result_port must not collide with any worker port"

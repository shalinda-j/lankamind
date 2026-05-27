"""
tests/test_coordinator.py
--------------------------
Unit tests for network.coordinator.NetworkCoordinator
"""
import pytest
import threading
import time

from network.coordinator import NetworkCoordinator, BASE_WORKER_PORT, RESULT_PORT


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def coord3():
    """Three-shard coordinator."""
    return NetworkCoordinator(num_shards=3, model="test-model", result_host="192.168.1.1")


@pytest.fixture
def coord1():
    """Single-shard coordinator."""
    return NetworkCoordinator(num_shards=1, model="tiny-model")


# ── Construction ──────────────────────────────────────────────────────────────

def test_invalid_num_shards():
    with pytest.raises(ValueError):
        NetworkCoordinator(num_shards=0, model="x")


def test_init_values(coord3):
    assert coord3.num_shards == 3
    assert coord3.model == "test-model"
    assert not coord3.all_registered
    assert not coord3.all_ready


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_assigns_shard_indices(coord3):
    r0 = coord3.register("10.0.0.1")
    r1 = coord3.register("10.0.0.2")
    r2 = coord3.register("10.0.0.3")
    assert r0["shard_idx"] == 0
    assert r1["shard_idx"] == 1
    assert r2["shard_idx"] == 2


def test_register_returns_model(coord3):
    r = coord3.register("10.0.0.1")
    assert r["model"] == "test-model"
    assert r["num_shards"] == 3


def test_register_assigns_correct_ports(coord3):
    r0 = coord3.register("10.0.0.1")
    r1 = coord3.register("10.0.0.2")
    r2 = coord3.register("10.0.0.3")
    assert r0["input_port"] == BASE_WORKER_PORT + 0
    assert r1["input_port"] == BASE_WORKER_PORT + 1
    assert r2["input_port"] == BASE_WORKER_PORT + 2


def test_register_tracks_remaining_slots(coord3):
    r0 = coord3.register("10.0.0.1")
    assert r0["slots_remaining"] == 2
    r1 = coord3.register("10.0.0.2")
    assert r1["slots_remaining"] == 1
    r2 = coord3.register("10.0.0.3")
    assert r2["slots_remaining"] == 0


def test_register_network_full(coord3):
    for i in range(3):
        coord3.register(f"10.0.0.{i}")
    r = coord3.register("10.0.0.99")
    assert r["error"] == "network_full"
    assert r["shard_idx"] == -1


def test_all_registered_flag(coord3):
    assert not coord3.all_registered
    coord3.register("A")
    assert not coord3.all_registered
    coord3.register("B")
    assert not coord3.all_registered
    coord3.register("C")
    assert coord3.all_registered


def test_single_shard_coordinator(coord1):
    r = coord1.register("localhost")
    assert r["shard_idx"] == 0
    assert r["slots_remaining"] == 0
    assert coord1.all_registered


# ── Routing / config ──────────────────────────────────────────────────────────

def test_config_returns_none_while_incomplete(coord3):
    coord3.register("10.0.0.1")
    assert coord3.get_config(0) is None   # only 1 of 3 registered


def test_config_available_when_all_registered(coord3):
    coord3.register("10.0.0.1")
    coord3.register("10.0.0.2")
    coord3.register("10.0.0.3")
    cfg0 = coord3.get_config(0)
    assert cfg0 is not None
    assert cfg0["shard_idx"] == 0
    assert cfg0["input_port"] == BASE_WORKER_PORT


def test_config_output_address_routing(coord3):
    """Shard N's output_address must point to shard N+1's host:port."""
    coord3.register("10.0.0.1")   # shard 0
    coord3.register("10.0.0.2")   # shard 1
    coord3.register("10.0.0.3")   # shard 2

    cfg0 = coord3.get_config(0)
    cfg1 = coord3.get_config(1)
    cfg2 = coord3.get_config(2)

    # Shard 0 → shard 1
    assert cfg0["output_address"] == f"tcp://10.0.0.2:{BASE_WORKER_PORT + 1}"
    # Shard 1 → shard 2
    assert cfg1["output_address"] == f"tcp://10.0.0.3:{BASE_WORKER_PORT + 2}"
    # Shard 2 → result collector on coordinator
    assert cfg2["output_address"] == f"tcp://192.168.1.1:{RESULT_PORT}"


def test_config_same_machine_routing():
    """All shards on localhost — addresses use 127.0.0.1 as reported IP."""
    coord = NetworkCoordinator(num_shards=2, model="m", result_host="127.0.0.1")
    coord.register("127.0.0.1")
    coord.register("127.0.0.1")
    cfg0 = coord.get_config(0)
    assert "127.0.0.1" in cfg0["output_address"]


def test_config_unknown_shard_returns_none(coord3):
    coord3.register("A"); coord3.register("B"); coord3.register("C")
    assert coord3.get_config(99) is None


# ── Ready signalling ──────────────────────────────────────────────────────────

def test_mark_ready(coord3):
    for ip in ["A", "B", "C"]:
        coord3.register(ip)
    assert not coord3.all_ready
    coord3.mark_ready(0)
    assert not coord3.all_ready
    coord3.mark_ready(1)
    assert not coord3.all_ready
    coord3.mark_ready(2)
    assert coord3.all_ready


def test_mark_ready_unknown_shard(coord3):
    ok = coord3.mark_ready(99)
    assert not ok


# ── Topology ──────────────────────────────────────────────────────────────────

def test_topology_structure(coord3):
    coord3.register("A")
    topo = coord3.topology()
    assert topo["num_shards"] == 3
    assert topo["registered"] == 1
    assert topo["slots_remaining"] == 2
    assert not topo["complete"]
    assert len(topo["workers"]) == 1
    assert topo["workers"][0]["shard_idx"] == 0
    assert topo["workers"][0]["host"] == "A"


def test_topology_complete_flag(coord3):
    for ip in ["A", "B", "C"]:
        coord3.register(ip)
    for i in range(3):
        coord3.mark_ready(i)
    topo = coord3.topology()
    assert topo["complete"]
    assert topo["registered"] == 3
    assert topo["slots_remaining"] == 0


def test_topology_workers_sorted_by_shard(coord3):
    coord3.register("C")  # gets shard 0
    coord3.register("B")  # gets shard 1
    coord3.register("A")  # gets shard 2
    topo = coord3.topology()
    indices = [w["shard_idx"] for w in topo["workers"]]
    assert indices == sorted(indices)


# ── Thread safety ─────────────────────────────────────────────────────────────

def test_concurrent_registration():
    """100 concurrent registrations — only num_shards succeed, no races."""
    N = 5
    coord = NetworkCoordinator(num_shards=N, model="x")
    results = []
    lock = threading.Lock()

    def reg(ip):
        r = coord.register(ip)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=reg, args=(f"10.{i}.0.1",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly N successful registrations
    successful = [r for r in results if r.get("shard_idx", -1) >= 0]
    assert len(successful) == N

    # No duplicate shard indices
    assigned = [r["shard_idx"] for r in successful]
    assert len(set(assigned)) == N

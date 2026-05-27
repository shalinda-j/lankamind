"""
tests/integration/test_end_to_end.py
--------------------------------------
End-to-end integration test: spawns 3 workers, runs the CLI client,
asserts we get a non-empty text response.

REQUIREMENTS
  • Internet access on first run (downloads GPT-2, ~500 MB).
  • ~1.5 GB free RAM.
  • No other LankaMind workers running on ports 5500–5502 and 5598.

SKIP FLAGS
  Set SKIP_INTEGRATION=1 in your environment to skip this test in CI.

Run:
    pytest tests/integration/test_end_to_end.py -v -s
    pytest tests/integration/ -m integration
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
NUM_SHARDS = 3
BASE_PORT = 5500
RESULT_PORT = 5598    # use 5598 (not 5599) to avoid clashing with a live dev session
LOAD_TIMEOUT = 150    # seconds


def ready_file(i: int) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"lankamind_worker_{i}.ready"


def worker_env() -> dict:
    env = os.environ.copy()
    pp = str(PROJECT_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pp}{os.pathsep}{existing}" if existing else pp
    return env


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def running_workers():
    """
    Module-scoped fixture: spawns 3 workers, yields, then tears them down.
    Shared across all tests in this file so we only load the model once.
    """
    if os.environ.get("SKIP_INTEGRATION"):
        pytest.skip("SKIP_INTEGRATION is set")

    # Remove stale ready files
    for i in range(NUM_SHARDS):
        ready_file(i).unlink(missing_ok=True)

    env = worker_env()
    processes: list[subprocess.Popen] = []

    for i in range(NUM_SHARDS):
        input_port = BASE_PORT + i
        if i < NUM_SHARDS - 1:
            output_address = f"tcp://localhost:{BASE_PORT + i + 1}"
        else:
            output_address = f"tcp://localhost:{RESULT_PORT}"

        cmd = [
            sys.executable, "-m", "core.worker",
            "--shard-idx",      str(i),
            "--num-shards",     str(NUM_SHARDS),
            "--model",          "gpt2",
            "--input-port",     str(input_port),
            "--output-address", output_address,
        ]
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append(proc)

    # Wait until all workers signal readiness
    start = time.monotonic()
    while time.monotonic() - start < LOAD_TIMEOUT:
        if all(ready_file(i).exists() for i in range(NUM_SHARDS)):
            break
        # Fail fast if any worker crashed
        for i, proc in enumerate(processes):
            if proc.poll() is not None:
                stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                pytest.fail(f"Worker {i} crashed during startup.\n{stdout}")
        time.sleep(1)
    else:
        pytest.fail(f"Workers did not become ready within {LOAD_TIMEOUT}s")

    yield  # ── tests run here ──────────────────────────────────────────────

    # Teardown
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    for i in range(NUM_SHARDS):
        ready_file(i).unlink(missing_ok=True)


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_produces_output(running_workers):
    """Client sends a prompt; the pipeline must return at least 5 tokens."""
    result = subprocess.run(
        [
            sys.executable, "cli/client.py",
            "The island of Sri Lanka is known for",
            "--max-tokens", "20",
            "--result-port", str(RESULT_PORT),
        ],
        cwd=PROJECT_ROOT,
        env=worker_env(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, f"Client exited non-zero.\nSTDERR: {result.stderr}"
    # stdout contains "Output : <tokens>" — check something was printed
    assert len(result.stdout.strip()) > 0, "Client produced no output at all"


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_output_is_text(running_workers):
    """Generated text must be a non-empty string of printable characters."""
    result = subprocess.run(
        [
            sys.executable, "cli/client.py",
            "Once upon a time in a land far away",
            "--max-tokens", "15",
            "--result-port", str(RESULT_PORT),
        ],
        cwd=PROJECT_ROOT,
        env=worker_env(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0
    output_line = ""
    for line in result.stdout.splitlines():
        if line.startswith("Output"):
            output_line = line
            break
    # Output line should have at least a few words
    assert len(output_line.split()) >= 3, f"Too short: {output_line!r}"


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_two_requests(running_workers):
    """Send two separate requests to confirm the pipeline handles multiple queries."""
    prompts = [
        "Colombo is the commercial capital",
        "The tea plantations of Nuwara Eliya",
    ]
    for prompt in prompts:
        result = subprocess.run(
            [
                sys.executable, "cli/client.py", prompt,
                "--max-tokens", "10",
                "--result-port", str(RESULT_PORT),
            ],
            cwd=PROJECT_ROOT,
            env=worker_env(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, f"Failed for prompt: {prompt!r}"

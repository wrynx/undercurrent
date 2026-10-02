"""The v0.1 GPU setup: .github/workflows/gpu.yml runs only on manual dispatch
(there is no GPU runner), and scripts/gpu_check.sh is the pre-release GPU
check that replaces the nightly run (RELEASING.md).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "gpu.yml"
SCRIPT = ROOT / "scripts" / "gpu_check.sh"
LIVE_LLAMA_TEST = "tests/examples/content_safety/test_live_llama_integration.py"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture(scope="module")
def script_text() -> str:
    if not SCRIPT.exists():  # e.g. running from an sdist
        pytest.skip("scripts/gpu_check.sh not present")
    return SCRIPT.read_text()


def test_gpu_workflow_is_dispatch_only() -> None:
    if not WORKFLOW.exists():
        pytest.skip("gpu.yml not present")
    wf = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML parses the bare key `on` as True.
    assert list(wf[True]) == ["workflow_dispatch"]


def test_script_runs_the_same_tests_as_the_workflow(script_text: str) -> None:
    assert "-m gpu" in script_text
    assert f"--ignore={LIVE_LLAMA_TEST}" in script_text
    assert '-m "not gpu" tests/adapters/vllm' in script_text
    assert "RUN_NETWORK_TESTS=1" in script_text


def test_script_requires_the_residual_stream_equivalence_test(script_text: str) -> None:
    assert "test_residual_stream_gpu" in script_text
    assert (ROOT / "tests" / "adapters" / "vllm" / "test_residual_stream_gpu.py").exists()


@needs_bash
def test_script_is_executable_and_parses(script_text: str) -> None:
    assert os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@needs_bash
def test_help(script_text: str) -> None:
    out = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "Manual GPU check" in out.stdout
    assert "--allow-unsupported-vllm" in out.stdout


@needs_bash
def test_unknown_argument_fails(script_text: str) -> None:
    out = subprocess.run(["bash", str(SCRIPT), "--nope"], capture_output=True, text=True)
    assert out.returncode != 0
    assert "unknown argument: --nope" in out.stderr


@needs_bash
def test_live_llama_requires_model_env(script_text: str) -> None:
    env = {k: v for k, v in os.environ.items() if k != "UNDERCURRENT_LIVE_LLAMA_MODEL"}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--python", sys.executable, "--live-llama"], capture_output=True, text=True, env=env
    )
    assert out.returncode != 0
    assert "UNDERCURRENT_LIVE_LLAMA_MODEL" in out.stderr


@needs_bash
def test_fails_clearly_without_cuda(script_text: str) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; this checks the no-GPU failure path")
    out = subprocess.run(["bash", str(SCRIPT), "--python", sys.executable], capture_output=True, text=True, timeout=300)
    assert out.returncode != 0
    assert "CUDA" in out.stderr
    # It stopped before running any tests.
    assert "pytest -m gpu" not in out.stdout

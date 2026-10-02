"""Repo-wide pytest configuration.

Markers (registered in ``pyproject.toml``; ``--strict-markers`` is on):

- ``gpu``: needs CUDA. Skipped automatically unless torch is importable and
  ``torch.cuda.is_available()`` is True, so a machine with vLLM installed but
  no GPU skips them instead of failing.
- ``network``: downloads from the Hugging Face Hub or the internet. Skipped
  unless you opt in with ``RUN_NETWORK_TESTS=1``.
- ``slow``: takes more than ~10s. Runs by default; deselect with
  ``-m "not slow"``.

``gpu`` tests also get a hard time limit (``GPU_TEST_TIMEOUT_S``, override with
``UNDERCURRENT_GPU_TEST_TIMEOUT``) when pytest-timeout is installed (it's in the
``dev`` extra): a stuck vLLM engine dumps every thread's stack and fails the
run instead of hanging it.

The default CPU-only, offline run is ``pytest -m "not gpu and not network"``
(plain ``pytest`` gives the same result here, via the skips above).

It also puts ``examples/content_safety/`` on ``sys.path`` so tests can import
the demo package ``content_safety_demo``, which is not installed.
"""

import os
import sys
from pathlib import Path

import pytest

# The content-safety demo lives in examples/ (not installed); both
# tests/examples/content_safety/ and the HF adapter tests import it.
_CONTENT_SAFETY_DIR = str(Path(__file__).resolve().parents[1] / "examples" / "content_safety")
if _CONTENT_SAFETY_DIR not in sys.path:
    sys.path.insert(0, _CONTENT_SAFETY_DIR)

RUN_NETWORK_TESTS_ENV = "RUN_NETWORK_TESTS"
GPU_TEST_TIMEOUT_S = 300


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _network_tests_enabled() -> bool:
    return os.environ.get(RUN_NETWORK_TESTS_ENV, "").strip().lower() in {"1", "true", "yes"}


def pytest_collection_modifyitems(config, items):
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if gpu_items and not _cuda_available():
        skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU (torch missing or CUDA unavailable)")
        for item in gpu_items:
            item.add_marker(skip_gpu)
    elif gpu_items and config.pluginmanager.hasplugin("timeout"):
        # method="thread": a hang inside C code (CUDA, ZeroMQ) never returns to the
        # interpreter, so a signal-based timeout couldn't interrupt it. The thread
        # method dumps all stacks and exits the run; scripts/gpu_check.sh reports it.
        seconds = int(os.environ.get("UNDERCURRENT_GPU_TEST_TIMEOUT", GPU_TEST_TIMEOUT_S))
        for item in gpu_items:
            if item.get_closest_marker("timeout") is None:
                item.add_marker(pytest.mark.timeout(seconds, method="thread"))

    network_items = [item for item in items if "network" in item.keywords]
    if network_items and not _network_tests_enabled():
        skip_network = pytest.mark.skip(reason=f"needs network access; set {RUN_NETWORK_TESTS_ENV}=1 to run")
        for item in network_items:
            item.add_marker(skip_network)

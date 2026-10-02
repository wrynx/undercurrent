"""The runtime vLLM version gate in `undercurrent.adapters.vllm.version_check`."""

from __future__ import annotations

import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from undercurrent.adapters.vllm import VLLMAdapterLimitationError, VLLMEngineAdapter, version_check
from undercurrent.adapters.vllm.version_check import ALLOW_UNSUPPORTED_VLLM_ENV, SUPPORTED_VLLM, check_vllm_version
from undercurrent.adapters.vllm.worker_extension import ProbingWorkerExtension

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def installed_vllm(monkeypatch):
    """Set the vLLM version the gate sees (None = not installed)."""
    monkeypatch.delenv(ALLOW_UNSUPPORTED_VLLM_ENV, raising=False)

    def _set(version):
        monkeypatch.setattr(version_check, "_installed_vllm_version", lambda: version)

    return _set


def test_env_var_name_follows_import_name():
    assert ALLOW_UNSUPPORTED_VLLM_ENV == "UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM"


@pytest.mark.parametrize("version", ["0.28.0", "0.28.3", "0.28.0+cu129", "0.28.1.dev7+g1234abc", "0.28.1rc1"])
def test_supported_versions_pass(installed_vllm, version):
    installed_vllm(version)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_vllm_version() == version


@pytest.mark.parametrize("version", ["0.27.9", "0.28.0rc1", "0.29.0", "0.29.0.dev1", "1.0.0", "not-a-version"])
def test_unsupported_versions_raise_actionable_error(installed_vllm, version):
    installed_vllm(version)
    with pytest.raises(VLLMAdapterLimitationError) as excinfo:
        check_vllm_version()
    message = str(excinfo.value)
    assert f"vllm=={version}" in message
    assert SUPPORTED_VLLM in message
    assert f"{ALLOW_UNSUPPORTED_VLLM_ENV}=1" in message
    assert "existing vLLM environment" in message
    assert 'pip install "undercurrent[vllm]"' in message


def test_override_downgrades_error_to_warning(installed_vllm, monkeypatch):
    installed_vllm("0.29.0")
    monkeypatch.setenv(ALLOW_UNSUPPORTED_VLLM_ENV, "1")
    with pytest.warns(RuntimeWarning, match=re.escape("vllm==0.29.0")):
        assert check_vllm_version() == "0.29.0"


def test_override_must_be_truthy(installed_vllm, monkeypatch):
    installed_vllm("0.29.0")
    monkeypatch.setenv(ALLOW_UNSUPPORTED_VLLM_ENV, "0")
    with pytest.raises(VLLMAdapterLimitationError):
        check_vllm_version()


def test_missing_vllm_is_left_to_require_vllm(installed_vllm):
    installed_vllm(None)
    assert check_vllm_version() is None
    VLLMEngineAdapter()  # constructing without vLLM still works; load_model() reports it


def test_adapter_construction_runs_the_gate(installed_vllm):
    installed_vllm("0.27.0")
    with pytest.raises(VLLMAdapterLimitationError, match=re.escape("vllm==0.27.0")):
        VLLMEngineAdapter()


def test_worker_extension_construction_runs_the_gate(installed_vllm):
    installed_vllm("0.29.0")
    with pytest.raises(VLLMAdapterLimitationError, match=re.escape("vllm==0.29.0")):
        ProbingWorkerExtension()


def test_worker_extension_rpc_entry_runs_the_gate(installed_vllm):
    # vLLM's mixin mechanism may skip __init__; _ensure_state() is the real entry point.
    installed_vllm("0.29.0")
    ext = ProbingWorkerExtension.__new__(ProbingWorkerExtension)
    with pytest.raises(VLLMAdapterLimitationError):
        ext._ensure_state()


def test_importing_the_adapter_does_not_import_vllm():
    code = "import sys, undercurrent.adapters.vllm; assert 'vllm' not in sys.modules, 'vllm imported'"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)


def test_supported_range_matches_pyproject_extra():
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extra = pyproject["project"]["optional-dependencies"]["vllm"]
    assert extra == [f"vllm{SUPPORTED_VLLM}"]


def test_compatibility_doc_names_the_supported_range():
    doc = (REPO_ROOT / "docs" / "compatibility.md").read_text()
    assert f"vllm{SUPPORTED_VLLM}" in doc.replace(" ", "")
    assert ALLOW_UNSUPPORTED_VLLM_ENV in doc

"""The lazy optional-dependency helpers in `undercurrent.adapters._optional`."""

from __future__ import annotations

import pytest

from undercurrent.adapters import _optional
from undercurrent.adapters._optional import MissingDependencyError


def _missing(name: str):
    raise ModuleNotFoundError(f"No module named {name!r}", name=name)


@pytest.mark.parametrize(
    "helper",
    [_optional.require_torch, _optional.require_transformers, _optional.require_vllm],
)
def test_missing_dependency_raises_missing_dependency_error(monkeypatch, helper):
    monkeypatch.setattr(_optional, "_import", _missing)
    with pytest.raises(MissingDependencyError) as excinfo:
        helper()
    assert isinstance(excinfo.value, ImportError)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_missing_vllm_message_names_both_install_routes(monkeypatch):
    monkeypatch.setattr(_optional, "_import", _missing)
    with pytest.raises(MissingDependencyError) as excinfo:
        _optional.require_vllm()
    message = str(excinfo.value)
    assert "existing vLLM environment" in message
    assert 'pip install "undercurrent[vllm]"' in message


def test_vllm_adapter_load_model_surfaces_missing_vllm(monkeypatch):
    from undercurrent.adapters.vllm import VLLMEngineAdapter

    real_import = _optional._import
    monkeypatch.setattr(_optional, "_import", lambda name: _missing(name) if name == "vllm" else real_import(name))
    with pytest.raises(MissingDependencyError, match="existing vLLM environment"):
        VLLMEngineAdapter().load_model("some-model")

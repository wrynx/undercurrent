"""Lazy imports for the heavy, environment-specific adapter dependencies.

Nothing in `undercurrent.adapters` imports torch, transformers or vllm at
module import time. Call `require_torch()` / `require_transformers()` /
`require_vllm()` inside the functions that actually need them, so
`import undercurrent` stays light and pure-Python modules such as
`undercurrent.adapters.vllm.seq_mapper` work without vLLM installed.

torch and transformers are base dependencies of `undercurrent`, so a missing
one means a broken install. vLLM is never installed by default, so a missing
vLLM is the one expected situation.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from ..errors import ProbingError


class MissingDependencyError(ProbingError, ImportError):
    """Raised when an optional package Undercurrent needs isn't installed (most often vLLM).

    The message says what to install. A distinct ``ImportError`` subclass, so
    you can catch it without also swallowing an unrelated import error inside
    torch, transformers or vLLM.
    """


#: Where the supported vLLM versions are documented (repo path and published page).
COMPATIBILITY_DOC = "docs/compatibility.md (https://wrynx.github.io/undercurrent/compatibility/)"


def vllm_missing_message(feature: str = "undercurrent.adapters.vllm") -> str:
    """The one message for "vLLM isn't installed", naming what needed it."""
    return (
        f"{feature} needs vLLM, which undercurrent never installs by default. vLLM must be installed in this "
        "environment, built for your CUDA and torch. Either install undercurrent into an existing vLLM "
        "environment or image (`pip install undercurrent` there), or install the tested vLLM range with "
        f'`pip install "undercurrent[vllm]"`. See {COMPATIBILITY_DOC} for the supported versions.'
    )


def _import(name: str) -> ModuleType:
    return importlib.import_module(name)


def require_torch() -> ModuleType:
    """Import and return the real `torch` module, or raise a clear error."""
    try:
        return _import("torch")
    except ImportError as exc:
        raise MissingDependencyError(
            "undercurrent's model adapters need `torch` for real generation (hook installation, "
            "tensor capture). torch is a base dependency of `undercurrent`; reinstall it "
            "(`pip install undercurrent`), or use a torch build matched to your CUDA/ROCm toolchain."
        ) from exc


def require_transformers() -> ModuleType:
    """Import and return the real `transformers` module, or raise a clear error."""
    try:
        return _import("transformers")
    except ImportError as exc:
        raise MissingDependencyError(
            "undercurrent.adapters.hf needs `transformers` for real generation. It is a base "
            "dependency of `undercurrent`; reinstall it: pip install undercurrent"
        ) from exc


def require_vllm(feature: str = "undercurrent.adapters.vllm") -> ModuleType:
    """Import and return the real `vllm` module, or raise a clear error.

    `feature` names what needs vLLM in the message (e.g. "backend='vllm'").
    """
    try:
        return _import("vllm")
    except ImportError as exc:
        raise MissingDependencyError(vllm_missing_message(feature)) from exc


__all__ = [
    "MissingDependencyError",
    "require_torch",
    "require_transformers",
    "require_vllm",
    "vllm_missing_message",
]

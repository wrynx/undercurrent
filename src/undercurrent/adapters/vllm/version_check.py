"""Runtime vLLM version gate.

The vLLM adapter hooks undocumented vLLM internals (the V1 model-runner
layout read by `introspection.py`, the `worker_extension_cls` mixin
mechanism, request-id handling), which can change in any vLLM minor
release. `introspection.py` fails fast when those internal *shapes* don't
match; this module fails fast, earlier and with a clearer message, when the
installed vLLM *version* is outside the range the adapter was validated
against.

vLLM is never installed by default, and the documented production path is
installing undercurrent into an existing vLLM environment or image -- so
pip's resolver usually never sees the `undercurrent[vllm]` range, and this
check is the primary guard.

The check reads the installed distribution's metadata
(`importlib.metadata.version("vllm")`); it never imports vllm itself. It
runs when `VLLMEngineAdapter` or `ProbingWorkerExtension` is constructed,
not at import time. If vLLM isn't installed at all it does nothing:
`load_model()` reports a missing vLLM through `require_vllm()`.

Internal: not part of the public API (see docs/api-stability.md) and may
change in any release.
"""

from __future__ import annotations

import os
import warnings
from importlib import metadata

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .._optional import COMPATIBILITY_DOC

# Keep in sync with the `vllm` extra in pyproject.toml and with
# docs/compatibility.md. Widen only after the GPU CI passes against the new
# minor.
SUPPORTED_VLLM = ">=0.28,<0.29"

ALLOW_UNSUPPORTED_VLLM_ENV = "UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM"

_TRUTHY = {"1", "true", "yes", "on"}


def _installed_vllm_version() -> str | None:
    """The installed vLLM distribution's version string, or None if vLLM
    isn't installed. Split out so tests can monkeypatch it."""
    try:
        return metadata.version("vllm")
    except metadata.PackageNotFoundError:
        return None


def _is_supported(version: str) -> bool:
    try:
        parsed = Version(version)
    except InvalidVersion:
        return False
    # prereleases=True so dev/rc builds *of a supported minor* (e.g.
    # 0.28.1.dev3) pass; `<0.29` still excludes 0.29 pre-releases (PEP 440).
    return SpecifierSet(SUPPORTED_VLLM).contains(parsed, prereleases=True)


def _override_enabled() -> bool:
    return os.environ.get(ALLOW_UNSUPPORTED_VLLM_ENV, "").strip().lower() in _TRUTHY


def check_vllm_version() -> str | None:
    """Check the installed vLLM against `SUPPORTED_VLLM`.

    Returns the installed version (None if vLLM isn't installed). Raises
    `VLLMAdapterLimitationError` if the version is unsupported, unless
    `UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM=1` is set, in which case it warns
    (`RuntimeWarning`) instead.
    """
    installed = _installed_vllm_version()
    if installed is None or _is_supported(installed):
        return installed

    message = (
        f"undercurrent.adapters.vllm supports vllm{SUPPORTED_VLLM}, but vllm=={installed} is installed. "
        "The adapter hooks vLLM internals that change between minor releases, so other versions "
        "may mis-capture or fail. Either install undercurrent into an existing vLLM environment or "
        f"image whose vLLM is in that range, or install the tested range with "
        f'`pip install "undercurrent[vllm]"`. See {COMPATIBILITY_DOC}. To try this vLLM version '
        f"anyway, set {ALLOW_UNSUPPORTED_VLLM_ENV}=1 (this turns the error into a warning)."
    )
    if _override_enabled():
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return installed

    # Imported here: adapter.py imports this module at load time.
    from .adapter import VLLMAdapterLimitationError

    raise VLLMAdapterLimitationError(message)


__all__ = ["ALLOW_UNSUPPORTED_VLLM_ENV", "SUPPORTED_VLLM", "check_vllm_version"]

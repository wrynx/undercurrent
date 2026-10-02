"""Undercurrent: Wrynx's activation-probing platform.

See the undercurrent before it surfaces. Load a model, attach probes, generate::

    from undercurrent import ProbedModel, probe

    @probe("norm")  # a stateless probe: activation in, score out
    def norm(record) -> float:
        return float(record.tensor.norm())

    spec = {"extraction_points": [{"name": "last", "layers": 0, "tensor_type": "residual_stream",
            "position": "prompt[-1]", "probe_type": "norm", "probe_kind": "single_shot"}]}
    with ProbedModel.from_pretrained("gpt2", spec=spec) as model:
        out = model.generate("The weather today is", max_new_tokens=20)
    print(out.text, out.probe_results["last"].verdict)

The names in ``__all__`` come in two tiers (see docs/api-stability.md):

- **Front door**: `ProbedModel`, `GenerationOutput`, `probe`,
  `register_probe`, `Probe`, `ExtractionPoint`, `load_spec`, `ProbingError`
  and `__version__`. Everything you need to probe a model.
- **Advanced**: the `Router`, probe data types, sinks, metrics, policy enums
  and the `EngineAdapter` ABC, for embedding Undercurrent in your own serving
  stack.

Names are loaded lazily on first access, so ``import undercurrent`` is fast
and never imports torch, transformers or vLLM. The concrete engine adapters
live in `undercurrent.adapters.hf` and `undercurrent.adapters.vllm`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

try:
    from ._version import __version__
except ImportError:  # e.g. running from a source tree without a build
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("undercurrent")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from .adapters import EngineAdapter, MissingDependencyError
    from .core import (
        Probe,
        ProbeAction,
        ProbeFactory,
        ProbeNotFoundError,
        ProbeResult,
        ProbeSignal,
        RequestContext,
        probe,
        register_probe,
    )
    from .errors import ProbingError
    from .model import GenerationOutput, ProbedModel
    from .router import (
        InMemoryMetricsRegistry,
        MetricsSink,
        MetricsSnapshot,
        OverflowPolicy,
        RequestHandle,
        Router,
        RouterError,
    )
    from .sinks import FileLogSink, LogSink, WebhookLogSink, chain, drop_keys, redact_keys
    from .spec import (
        ActivationRecord,
        ExecutionMode,
        ExtractionPoint,
        InterventionMode,
        InterventionPolicy,
        ProbeKind,
        ProbeSpec,
        SpecValidationError,
        TensorType,
        TimeoutAction,
        load_spec,
        parse_position,
    )

#: Public name -> the submodule it is loaded from (relative to this package).
_LAZY: dict[str, str] = {
    # Front door
    "ProbedModel": ".model",
    "GenerationOutput": ".model",
    "probe": ".core",
    "register_probe": ".core",
    "Probe": ".core",
    "ExtractionPoint": ".spec",
    "load_spec": ".spec",
    "ProbingError": ".errors",
    # Advanced: probes and data types
    "ProbeFactory": ".core",
    "ProbeSignal": ".core",
    "ProbeAction": ".core",
    "ProbeResult": ".core",
    "RequestContext": ".core",
    "ActivationRecord": ".spec",
    "ProbeSpec": ".spec",
    "parse_position": ".spec",
    "TensorType": ".spec",
    "ProbeKind": ".spec",
    "ExecutionMode": ".spec",
    "InterventionPolicy": ".spec",
    "InterventionMode": ".spec",
    "TimeoutAction": ".spec",
    # Advanced: router and metrics
    "Router": ".router",
    "RequestHandle": ".router",
    "OverflowPolicy": ".router",
    "MetricsSink": ".router",
    "InMemoryMetricsRegistry": ".router",
    "MetricsSnapshot": ".router",
    # Advanced: observation sinks and redaction
    "LogSink": ".sinks",
    "FileLogSink": ".sinks",
    "WebhookLogSink": ".sinks",
    "redact_keys": ".sinks",
    "drop_keys": ".sinks",
    "chain": ".sinks",
    # Advanced: engine adapters
    "EngineAdapter": ".adapters",
    # Advanced: errors worth catching by name
    "SpecValidationError": ".spec",
    "ProbeNotFoundError": ".core",
    "RouterError": ".router",
    "MissingDependencyError": ".adapters",
}

__all__ = [
    # Front door
    "ProbedModel",
    "GenerationOutput",
    "probe",
    "register_probe",
    "Probe",
    "ExtractionPoint",
    "load_spec",
    "ProbingError",
    "__version__",
    # Advanced
    "ActivationRecord",
    "EngineAdapter",
    "ExecutionMode",
    "FileLogSink",
    "InMemoryMetricsRegistry",
    "InterventionMode",
    "InterventionPolicy",
    "LogSink",
    "MetricsSink",
    "MetricsSnapshot",
    "MissingDependencyError",
    "OverflowPolicy",
    "ProbeAction",
    "ProbeFactory",
    "ProbeKind",
    "ProbeNotFoundError",
    "ProbeResult",
    "ProbeSignal",
    "ProbeSpec",
    "RequestContext",
    "RequestHandle",
    "Router",
    "RouterError",
    "SpecValidationError",
    "TensorType",
    "TimeoutAction",
    "WebhookLogSink",
    "chain",
    "drop_keys",
    "parse_position",
    "redact_keys",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value  # later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

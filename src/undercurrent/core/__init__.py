"""undercurrent.core: the probe interface layer for the activation-probing platform.

This package defines *how a probe plugs into the platform* -- the Probe
lifecycle, the isolation model between spawned instances, and the signal /
result data types -- and nothing about *how activations get to a probe* or
*which probe runs where*. Those are the router's and the adapters' jobs,
built on top of this package (and of `undercurrent.spec`, which this package
depends on for `ActivationRecord`).

Typical usage (from the router's perspective, illustrative only -- the
router itself is not implemented in this package)::

    from undercurrent.core import Probe, ProbeAction, RequestContext

    probe = SomeProbeSubclass.spawn(request_id, extraction_point_name)
    probe.on_start(RequestContext(request_id, prompt_metadata, ep_config))
    for record in matching_activations:
        signal = probe.on_activation(record)
        if signal is not None and signal.action is ProbeAction.ABORT:
            break
    result = probe.on_end(request_ctx)

Public surface:
    - Lifecycle: `Probe`, `ProbeFactory`, `RequestContext`
    - Function probes: `probe()` turns ``fn(record) -> score`` into a
      single_shot Probe (see `undercurrent.core.function_probe`)
    - Lookup by name: `register_probe()`, `get_probe_factory()`,
      `list_probes()`, `ProbeRegistry` (see `undercurrent.core.registry`)
    - Data types: `ProbeSignal`, `ProbeAction`, `ProbeResult`
    - Example probes: `undercurrent.core.examples`

`ActivationRecord` is also importable from here for probe authors'
convenience, but its home (and its place in ``__all__``) is
`undercurrent.spec`.
"""

from .activation import ActivationRecord as ActivationRecord  # convenience alias, not in __all__
from .context import RequestContext
from .function_probe import FunctionProbe, probe
from .probe import Probe, ProbeFactory
from .registry import (
    ProbeNotFoundError,
    ProbeRegistry,
    default_registry,
    get_probe_factory,
    list_probes,
    register_probe,
    unregister_probe,
)
from .result import ProbeResult
from .signal import ProbeAction, ProbeSignal

__all__ = [
    "FunctionProbe",
    "Probe",
    "ProbeAction",
    "ProbeFactory",
    "ProbeNotFoundError",
    "ProbeRegistry",
    "ProbeResult",
    "ProbeSignal",
    "RequestContext",
    "default_registry",
    "get_probe_factory",
    "list_probes",
    "probe",
    "register_probe",
    "unregister_probe",
]

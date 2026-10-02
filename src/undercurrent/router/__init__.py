"""undercurrent.router: dispatches ActivationRecords to the correct probe
instance(s), per request, according to each extraction point's
execution_mode.

Built on `undercurrent.spec` (ExtractionPoint, ActivationRecord) and
`undercurrent.core` (Probe, ProbeFactory, ProbeSignal, ProbeResult,
RequestContext). Implements no inference-engine
adapter: the router is fully driven and tested via synthetic
ActivationRecord streams, with no real inference engine involved. A real
adapter is expected to call `route()` once per activation as it occurs
during inference.

Typical usage::

    from undercurrent.router import ProbeAction, ProbeFactory, Router

    with Router(probe_registry={"my_probe_type": ProbeFactory(MyProbe)}) as router:
        with router.request(extraction_points=spec, request_id=request_id) as req:
            for record in activation_stream:
                signal = req.route(record)   # non-None only for inline extraction points
                if signal is not None and signal.action is ProbeAction.ABORT:
                    break
        results = req.results   # {extraction_point_name: ProbeResult}

`router.request(...)` always calls `end_request` on exit, even if the body
raises. The lower-level `register_request` / `route` / `end_request` calls
remain available. When an engine adapter calls `end_request` itself, use
`router.on_request_end(listener)` to receive each request's results.

Public surface (the *advanced* API; most users want `undercurrent.ProbedModel`):
    - `Router`, `RequestHandle`, `RequestEndListener`,
      `RouterError`
    - Async execution: `OverflowPolicy`, `default_worker_pool_size()`,
      `DEFAULT_QUEUE_DEPTH`, `DEFAULT_DRAIN_TIMEOUT`,
      `DEFAULT_CIRCUIT_BREAKER_THRESHOLD`
    - Metrics: `MetricsSink`, `InMemoryMetricsRegistry`,
      `MetricsSnapshot`

The spec and probe types the router works with (`ExtractionPoint`,
`InterventionPolicy`, `ProbeFactory`, `ProbeAction`, ...) are also importable
from here for convenience, but they belong to `undercurrent.spec` and
`undercurrent.core` and are listed in those packages' ``__all__``. The
modules `binding`, `overflow` (apart from `OverflowPolicy`) and `metrics`
(apart from the three public types) are internal.
"""

# Convenience aliases for the spec/core types the router is configured with;
# their public home is undercurrent.spec / undercurrent.core.
from ..core import Probe, ProbeAction, ProbeFactory, ProbeResult, ProbeSignal, RequestContext  # noqa: F401
from ..spec import (  # noqa: F401
    ActivationRecord,
    ExecutionMode,
    ExtractionPoint,
    InterventionMode,
    InterventionPolicy,
    ProbeKind,
    TimeoutAction,
)
from .errors import RouterError
from .metrics import InMemoryMetricsRegistry, MetricsSink, MetricsSnapshot
from .overflow import OverflowPolicy
from .request import RequestHandle
from .router import (
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_DRAIN_TIMEOUT,
    DEFAULT_QUEUE_DEPTH,
    RequestEndListener,
    Router,
    default_worker_pool_size,
)

__all__ = [
    "DEFAULT_CIRCUIT_BREAKER_THRESHOLD",
    "DEFAULT_DRAIN_TIMEOUT",
    "DEFAULT_QUEUE_DEPTH",
    "InMemoryMetricsRegistry",
    "MetricsSink",
    "MetricsSnapshot",
    "OverflowPolicy",
    "RequestEndListener",
    "RequestHandle",
    "Router",
    "RouterError",
    "default_worker_pool_size",
]

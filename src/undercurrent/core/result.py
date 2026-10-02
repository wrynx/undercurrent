"""ProbeResult: the final output of a probe's lifecycle for one request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .signal import ProbeSignal


@dataclass
class ProbeResult:
    """Returned by `Probe.on_end`, exactly once per (request, extraction point).

    Attributes:
        request_id: the request this result belongs to.
        extraction_point_name: which extraction point produced it.
        verdict: the probe's conclusion. Deliberately untyped: a classifier
            probe might return a label + logits dict, a trajectory probe a
            running score, an intervention probe just a bool. Each probe
            should document its own verdict shape.
        signal_history: every ProbeSignal this probe emitted during the
            request's lifecycle, in emission order. May be empty if the
            probe never intervened.
        metadata: probe-defined auxiliary data that isn't part of the
            verdict itself (timing, debug info, etc.).
    """

    request_id: str
    extraction_point_name: str
    verdict: Any
    signal_history: list[ProbeSignal] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

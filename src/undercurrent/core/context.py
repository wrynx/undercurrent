"""RequestContext: what a probe knows about the request it's attached to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestContext:
    """What a probe knows about its request, passed to ``on_start`` and ``on_end``.

    Built by the router. Frozen: keep a probe's own mutable state on the
    probe instance (set in ``__init__`` or ``on_start``), not here.

    Attributes:
        request_id: identifier of the inference request.
        prompt_metadata: router/adapter-defined metadata about the prompt
            (e.g. model name, prompt length, sampling params). Its shape is
            deliberately open-ended.
        extraction_point_config: the resolved
            [`ExtractionPoint`][undercurrent.spec.ExtractionPoint] this probe
            was spawned for, so the probe can read its layer, tensor and
            position config.
    """

    request_id: str
    prompt_metadata: dict[str, Any]
    extraction_point_config: Any

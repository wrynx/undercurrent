"""Serialize a resolved `ProbeSpec` back to plain data / YAML.

Together with `undercurrent.spec.parser`, this makes the spec -> internal
representation -> spec round trip lossless: ``parse_dict(probe_spec_to_dict(spec))``
reproduces an equivalent ``ProbeSpec`` (equivalent, not necessarily
byte-identical -- e.g. ``prompt[-1]`` round-trips as the string
``"prompt[-1]"`` rather than as whatever whitespace the user originally wrote).
"""

from __future__ import annotations

from typing import Any

import yaml

from .resolved import ExtractionPoint, ProbeSpec


def extraction_point_to_dict(point: ExtractionPoint) -> dict[str, Any]:
    """Convert one resolved extraction point back to a plain (spec-shaped) dict."""
    data: dict[str, Any] = {
        "name": point.name,
        "layers": point.layers[0] if len(point.layers) == 1 else list(point.layers),
        "tensor_type": point.tensor_type.value,
        "position": point.position.to_raw(),
        "probe_type": point.probe_type,
        "probe_kind": point.probe_kind.value,
        "execution_mode": point.execution_mode.value,
    }
    if point.stride is not None:
        data["stride"] = point.stride
    if point.until is not None:
        data["until"] = point.until.value
    if point.queue_depth is not None:
        data["queue_depth"] = point.queue_depth
    if point.intervention is not None:
        intervention: dict[str, Any] = {"mode": point.intervention.mode.value}
        if point.intervention.timeout_ms is not None:
            intervention["timeout_ms"] = point.intervention.timeout_ms
        if point.intervention.on_timeout.value != "continue":
            intervention["on_timeout"] = point.intervention.on_timeout.value
        data["intervention"] = intervention
    if point.probe_args:
        data["probe_args"] = dict(point.probe_args)
    return data


def probe_spec_to_dict(spec: ProbeSpec) -> dict[str, Any]:
    """Convert a resolved ProbeSpec back to a plain (spec-shaped) dict."""
    return {
        "version": spec.version,
        "extraction_points": [extraction_point_to_dict(p) for p in spec.extraction_points],
    }


def to_yaml(spec: ProbeSpec) -> str:
    """Serialize a resolved ProbeSpec to a YAML string."""
    return yaml.safe_dump(probe_spec_to_dict(spec), sort_keys=False)

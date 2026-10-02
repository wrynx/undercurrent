"""Binds `undercurrent.spec.ExtractionPoint` config to this package's `Probe`
subclasses -- the wiring a router (or, eventually, a live inference-engine
adapter) uses to sanity-check a parsed spec against the probe it names
*before* ever spawning an instance for a real request.

This is the piece that makes this package a real end-to-end reference test
case rather than just two probes: `resolve_probe_cls` and
`probe_kwargs_from_extraction_point` catch the exact class of wiring bug
this platform needs to guard against (a spec naming a `probe_type` nothing
implements, a `probe_kind` mismatch between the spec and the probe class,
or a probe's `layer` config silently drifting from its extraction point's
own `layers`) at registration time, not partway through a request.
"""

from __future__ import annotations

from typing import Any

from undercurrent import ExtractionPoint, ProbeKind
from undercurrent.core import Probe, get_probe_factory

# Importing the probe modules registers them (via `@register_probe`) under
# the probe_type names the YAML specs use: content_safety_single_token,
# content_safety_trajectory and custom_mlp_safety_check.
from . import custom_mlp, single_token, trajectory  # noqa: F401


def resolve_probe_cls(point: ExtractionPoint) -> type[Probe]:
    """Look up and validate the Probe subclass a resolved ExtractionPoint names.

    Raises `undercurrent.core.ProbeNotFoundError` if nothing is registered
    for `point.probe_type`, and ValueError if `point.probe_kind` doesn't
    match what that probe class declares.
    """
    probe_cls = get_probe_factory(point.probe_type).probe_cls

    expected_kind = ProbeKind(probe_cls.probe_kind)
    if point.probe_kind != expected_kind:
        raise ValueError(
            f"extraction point {point.name!r}: probe_type={point.probe_type!r} implements "
            f"probe_kind={expected_kind.value!r}, but the spec declares "
            f"probe_kind={point.probe_kind.value!r}"
        )

    return probe_cls


def probe_kwargs_from_extraction_point(point: ExtractionPoint, **overrides: Any) -> dict[str, Any]:
    """Derive this package's constructor kwargs (`layer`, ...) from a
    resolved ExtractionPoint, so a probe's `layer` config can't silently
    drift from the extraction point's own `layers`. `overrides` (e.g.
    `threshold`, `seed`) come from probe-specific config the spec itself
    doesn't carry.
    """
    if len(point.layers) != 1:
        raise ValueError(
            f"extraction point {point.name!r}: content-safety probes are single-layer; got layers={point.layers}"
        )
    kwargs: dict[str, Any] = {"layer": point.layers[0]}
    kwargs.update(overrides)
    return kwargs

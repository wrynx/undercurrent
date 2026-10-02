"""Shared helper for the example probes: coerce a framework-agnostic tensor
into a flat list of floats without importing numpy/torch/jax."""

from __future__ import annotations

from typing import Any


def to_float_list(tensor: Any) -> list[float]:
    if hasattr(tensor, "tolist"):
        values = tensor.tolist()
    elif hasattr(tensor, "__iter__"):
        values = list(tensor)
    else:
        return [float(tensor)]

    flat: list[float] = []
    for v in values:
        if isinstance(v, (list, tuple)):
            flat.extend(float(x) for x in v)
        else:
            flat.append(float(v))
    return flat

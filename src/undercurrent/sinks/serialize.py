"""Convert arbitrary probe payloads (dataclasses, enums, tensors, ...) into
plain JSON-safe Python values.

`ProbeSignal.metadata` / `ProbeResult.verdict` / `ProbeResult.metadata` are
deliberately untyped at the `undercurrent.core` layer -- a real probe may stuff a
torch/numpy tensor, or anything else, into any of them. This module's job is
to make that always safe to log: known-safe values pass through unchanged,
dataclasses/enums are unwrapped structurally, and anything that looks like a
tensor (duck-typed via `.shape`/`.dtype`, so this has no hard dependency on
torch or numpy) is summarized instead of dumped -- logs should stay small
and grep-able, not carry gigabytes of raw activations.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def to_jsonable(value: Any) -> Any:
    """Recursively convert `value` into something `json.dumps` can handle.

    - dataclass instances -> dict of their fields, recursively converted.
    - `Enum` members -> `.value`.
    - dict -> dict with keys coerced to `str` (JSON object keys must be
      strings) and values recursively converted.
    - list/tuple/set -> list, recursively converted.
    - tensor-like objects (anything with both `.shape` and `.dtype`, e.g.
      a `torch.Tensor` or `numpy.ndarray`) -> a small summary dict, never
      the raw values.
    - `bytes`/`bytearray` -> summarized the same way as a tensor would be,
      since dumping raw bytes into a log line is just as undesirable.
    - anything else JSON-native (str/int/float/bool/None) -> passed through.
    - anything else -> ``str(value)``, as a last-resort fallback rather
      than raising.
    """
    if isinstance(value, _PRIMITIVE_TYPES):
        return value
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return {"__summary__": "bytes", "length": len(value)}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _summarize_tensor(value)
    try:
        return str(value)
    except Exception:  # noqa: BLE001 -- logging must never raise on a hostile __str__
        return f"<unserializable {type(value).__name__}>"


def _summarize_tensor(value: Any) -> dict[str, Any]:
    summary = {
        "__summary__": "tensor",
        "type": type(value).__name__,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    device = getattr(value, "device", None)
    if device is not None:
        summary["device"] = str(device)
    return summary

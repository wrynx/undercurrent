"""Shared shape for one logged observation, used by every concrete `LogSink`.

Internal: not part of the public API (see docs/api-stability.md) and may
change in any release.
"""

from __future__ import annotations

import time
from typing import Any

from .serialize import to_jsonable

SIGNAL = "signal"
RESULT = "result"


def build_log_record(kind: str, request_id: str, extraction_point_name: str, payload: Any) -> dict[str, Any]:
    """Build the plain-dict, JSON-safe record every sink logs.

    `kind` is `SIGNAL` or `RESULT`; `payload` is the `ProbeSignal` or
    `ProbeResult` being logged, run through `to_jsonable` so the result is
    always safe to `json.dumps` regardless of what a probe stuffed into its
    metadata/verdict fields.
    """
    return {
        "kind": kind,
        "request_id": request_id,
        "extraction_point_name": extraction_point_name,
        "timestamp": time.time(),
        "payload": to_jsonable(payload),
    }

"""ProbeSignal: the message a probe can emit in response to one activation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProbeAction(str, Enum):
    """What the caller (router/adapter) should do in response to a signal."""

    CONTINUE = "continue"
    """No intervention needed."""
    ABORT = "abort"
    """Stop generation (honored for inline extraction points)."""
    FLAG = "flag"
    """Record the activation as flagged; generation continues."""


@dataclass
class ProbeSignal:
    """Returned by ``Probe.on_activation`` when an activation warrants a message.

    Attributes:
        action: what the caller should do. Defaults to CONTINUE, i.e. "no
            intervention needed."
        metadata: probe-defined, e.g. scores or intermediate values useful
            for logging/debugging.
        confidence: optional scalar confidence in this signal, meaning is
            probe-defined (e.g. classifier probability, distance from a
            threshold).
        timestamp: unix timestamp (seconds) of when this signal was
            created. Defaults to the time of construction.
    """

    action: ProbeAction = ProbeAction.CONTINUE
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    timestamp: float = field(default_factory=time.time)

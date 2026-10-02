"""TrajectorySafetyProbe: a trajectory content-safety probe.

Subscribes to every matching activation for a request (typically
`generated[*]` at a fixed layer) and maintains a running safety score across
calls via `TrajectoryRecurrence` (a structural, random-initialized GRU-cell
recurrence -- see `models.py`). Emits `action=abort` the first time the
running score crosses `threshold`.
"""

from __future__ import annotations

from typing import Any

import torch

from undercurrent import (
    ActivationRecord,
    Probe,
    ProbeAction,
    ProbeResult,
    ProbeSignal,
    RequestContext,
    register_probe,
)

from .models import TrajectoryRecurrence
from .tensors import to_flat_tensor


@register_probe("content_safety_trajectory")
class TrajectorySafetyProbe(Probe):
    """trajectory probe: running GRU-recurrence score with abort-on-threshold.

    Verdict shape (set in `on_end`):
        {"final_score": float, "count": int, "aborted": bool, "score_history": List[float]}
    """

    probe_kind = "trajectory"

    def __init__(
        self,
        layer: int,
        threshold: float = 0.85,
        hidden_size: int = 16,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._layer = layer
        self._threshold = threshold
        self._hidden_size = hidden_size
        self._seed = seed

        self._recurrence: TrajectoryRecurrence | None = None
        self._hidden: torch.Tensor | None = None
        self._running_score: float = 0.0
        self._count: int = 0
        self._aborted: bool = False
        self._score_history: list[float] = []
        self._signal_history: list[ProbeSignal] = []

    def on_start(self, request_ctx: RequestContext) -> None:
        pass  # nothing to set up beyond what __init__ already initialized

    @property
    def running_score(self) -> float:
        return self._running_score

    def _build_recurrence(self, input_dim: int) -> TrajectoryRecurrence:
        """Construct the recurrence once the real activation's flattened
        dimension is known -- see `SingleTokenSafetyProbe._build_classifier`
        and `models.py`'s module docstring for why this is deferred rather
        than built eagerly with an `nn.Lazy*` layer.
        """
        if self._seed is None:
            return TrajectoryRecurrence(input_dim, self._hidden_size)
        with torch.random.fork_rng():
            torch.manual_seed(self._seed)
            return TrajectoryRecurrence(input_dim, self._hidden_size)

    def _check_intervention(self) -> ProbeSignal | None:
        """Called after each activation is folded into the running score.

        Emits action=abort the first time the running score crosses
        `self._threshold`, and nothing (None) afterward -- once aborted, a
        probe shouldn't keep re-emitting abort signals every subsequent
        call.
        """
        if self._aborted:
            return None
        if self._running_score > self._threshold:
            self._aborted = True
            return ProbeSignal(
                action=ProbeAction.ABORT,
                confidence=self._running_score,
                metadata={
                    "reason": "content_safety_threshold_exceeded",
                    "score": self._running_score,
                    "count": self._count,
                    "layer": self._layer,
                },
            )
        return None

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        if record.layer != self._layer:
            raise ValueError(
                f"{type(self).__name__} configured for layer={self._layer}, got "
                f"ActivationRecord.layer={record.layer} (extraction_point="
                f"{record.extraction_point_name!r})"
            )

        x = to_flat_tensor(record.tensor)
        if self._recurrence is None:
            self._recurrence = self._build_recurrence(x.shape[0])
            self._recurrence.eval()
            self._hidden = self._recurrence.initial_hidden()

        with torch.no_grad():
            score_tensor, self._hidden = self._recurrence(x, self._hidden)
            score = float(score_tensor)

        self._count += 1
        self._running_score = score
        self._score_history.append(score)

        signal = self._check_intervention()
        if signal is not None:
            self._signal_history.append(signal)
        return signal

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        verdict: dict[str, Any] = {
            "final_score": self._running_score,
            "count": self._count,
            "aborted": self._aborted,
            "score_history": list(self._score_history),
        }
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict=verdict,
            signal_history=list(self._signal_history),
            metadata={"threshold": self._threshold, "layer": self._layer},
        )

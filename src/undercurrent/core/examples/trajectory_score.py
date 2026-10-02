"""TrajectoryScoreProbe: a trajectory example probe.

Accumulates a running score across every matching activation for a
request, checking after each one whether the running mean has crossed an
intervention threshold. Exists to validate the `Probe` interface for
stateful, multi-activation probes -- `_score_activation` is a deterministic
placeholder standing in for a real per-token scoring model.
"""

from __future__ import annotations

from typing import Any

from ..activation import ActivationRecord
from ..context import RequestContext
from ..probe import Probe
from ..result import ProbeResult
from ..signal import ProbeAction, ProbeSignal
from ._util import to_float_list


class TrajectoryScoreProbe(Probe):
    """trajectory probe: running-mean score with abort-on-threshold.

    Scores each activation with a deterministic stub (the mean of its
    values) and emits an ``ABORT`` signal the first time the running mean
    reaches ``threshold``.

    The verdict (set in ``on_end``) is
    ``{"final_mean": float, "count": int, "aborted": bool}``.

    Args:
        threshold: running-mean score at which the probe asks to abort.
    """

    probe_kind = "trajectory"

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__()
        self._threshold = threshold
        self._running_sum: float = 0.0
        self._count: int = 0
        self._signal_history: list[ProbeSignal] = []
        self._aborted: bool = False

    def on_start(self, request_ctx: RequestContext) -> None:
        pass  # nothing to set up beyond what __init__ already initialized

    @property
    def running_mean(self) -> float:
        """Mean score of the activations seen so far (0.0 before any)."""
        return self._running_sum / self._count if self._count else 0.0

    def _score_activation(self, record: ActivationRecord) -> float:
        """Stub scalar score for one activation: mean of its values.

        Deterministic placeholder -- not a trained model.
        """
        values = to_float_list(record.tensor)
        return sum(values) / len(values) if values else 0.0

    def check_intervention(self) -> ProbeSignal | None:
        """Called after each activation is folded into the running score.

        Returns an ``ABORT`` signal the first time the running mean reaches
        ``threshold``, and None otherwise (including every call after
        that, so the abort is emitted once).
        """
        if self._aborted:
            return None
        if self.running_mean >= self._threshold:
            self._aborted = True
            return ProbeSignal(
                action=ProbeAction.ABORT,
                confidence=self.running_mean,
                metadata={"running_mean": self.running_mean, "count": self._count},
            )
        return None

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        score = self._score_activation(record)
        self._running_sum += score
        self._count += 1

        signal = self.check_intervention()
        if signal is not None:
            self._signal_history.append(signal)
        return signal

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        verdict: dict[str, Any] = {
            "final_mean": self.running_mean,
            "count": self._count,
            "aborted": self._aborted,
        }
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict=verdict,
            signal_history=list(self._signal_history),
            metadata={"threshold": self._threshold},
        )

"""MLPClassifierProbe: a single_shot example probe.

Classifies one activation into one of N classes via a stub forward pass.
Exists to validate the `Probe` interface end-to-end, not as a real model --
`_forward` is a deterministic placeholder standing in for a trained MLP
head; swap it out for a real model without touching the rest of the class.
"""

from __future__ import annotations

from typing import Any

from ..activation import ActivationRecord
from ..context import RequestContext
from ..probe import Probe
from ..result import ProbeResult
from ..signal import ProbeAction, ProbeSignal
from ._util import to_float_list


class MLPClassifierProbe(Probe):
    """single_shot probe: one activation in, one classification verdict out.

    Uses a deterministic stub in place of a trained MLP head.

    The verdict (set in ``on_end``) is
    ``{"predicted_class": int, "logits": list[float]}``.

    Args:
        num_classes: number of classes (logits) to produce.
    """

    probe_kind = "single_shot"

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._verdict: dict[str, Any] | None = None
        self._signal_history: list[ProbeSignal] = []

    def on_start(self, request_ctx: RequestContext) -> None:
        pass  # nothing to set up beyond what __init__ already initialized

    def _forward(self, values: list[float]) -> list[float]:
        """Stub MLP forward pass: values -> per-class logits.

        Deterministic placeholder (a fixed per-class linear projection of
        the mean activation value) -- not a trained model.
        """
        mean_activation = sum(values) / len(values) if values else 0.0
        return [mean_activation * (class_idx + 1) for class_idx in range(self._num_classes)]

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        logits = self._forward(to_float_list(record.tensor))
        predicted_class = max(range(len(logits)), key=lambda i: logits[i])

        self._verdict = {"predicted_class": predicted_class, "logits": logits}
        signal = ProbeSignal(
            action=ProbeAction.CONTINUE,
            confidence=logits[predicted_class],
            metadata={"predicted_class": predicted_class},
        )
        self._signal_history.append(signal)
        return signal

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict=self._verdict,
            signal_history=list(self._signal_history),
            metadata={"num_classes": self._num_classes},
        )

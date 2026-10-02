"""SingleTokenSafetyProbe: a single_shot content-safety probe.

Wraps `SafetyClassifierHead` (a structural, random-initialized linear/MLP
head -- see `models.py`) to classify one activation, from one layer/token,
into a safety score in [0, 1].
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

from .models import SafetyClassifierHead
from .tensors import to_flat_tensor


@register_probe("content_safety_single_token")
class SingleTokenSafetyProbe(Probe):
    """single_shot probe: one activation in, one safety verdict out.

    A single_shot probe finalizes in exactly one call -- so `on_activation`
    holds all the verdict logic (runs the classifier, applies the
    threshold, builds the final verdict dict) and `on_end` just packages
    whatever `on_activation` already computed into a `ProbeResult`. This
    keeps the two methods consistent with the `Probe` contract: `on_end`
    must always return a well-formed `ProbeResult`, even for a request
    where `on_activation` was never called (e.g. the extraction point never
    matched a token) -- handled below by falling back to an
    unflagged/`None`-score verdict in that case, rather than raising.

    Verdict shape (set in `on_activation`, packaged in `on_end`):
        {"score": Optional[float], "flagged": bool, "layer": int, "token_pos": Optional[int]}
    """

    probe_kind = "single_shot"

    def __init__(
        self,
        layer: int,
        threshold: float = 0.5,
        hidden_size: int = 32,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._layer = layer
        self._threshold = threshold
        self._hidden_size = hidden_size
        self._seed = seed

        self._classifier: SafetyClassifierHead | None = None
        self._verdict: dict[str, Any] | None = None
        self._signal_history: list[ProbeSignal] = []
        self._called = False

    def on_start(self, request_ctx: RequestContext) -> None:
        pass  # nothing to set up beyond what __init__ already initialized

    def _build_classifier(self, input_dim: int) -> SafetyClassifierHead:
        """Construct the classifier head once the real activation's flattened
        dimension is known. If `seed` was given, the construction (and hence
        its random weight init) is scoped inside a forked RNG state so it
        neither depends on, nor perturbs, the ambient global torch RNG --
        see `models.py`'s module docstring for why this can't just be a
        `nn.LazyLinear` seeded once up front.
        """
        if self._seed is None:
            return SafetyClassifierHead(input_dim, self._hidden_size)
        with torch.random.fork_rng():
            torch.manual_seed(self._seed)
            return SafetyClassifierHead(input_dim, self._hidden_size)

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        if self._called:
            raise RuntimeError(
                f"{type(self).__name__} (probe_kind='single_shot') received a second "
                "on_activation call for extraction_point="
                f"{self.extraction_point_name!r}; a single_shot probe must be wired to "
                "exactly one matching activation per request"
            )
        self._called = True

        if record.layer != self._layer:
            raise ValueError(
                f"{type(self).__name__} configured for layer={self._layer}, got "
                f"ActivationRecord.layer={record.layer} (extraction_point="
                f"{record.extraction_point_name!r})"
            )

        x = to_flat_tensor(record.tensor)
        if self._classifier is None:
            self._classifier = self._build_classifier(x.shape[0])
            self._classifier.eval()

        with torch.no_grad():
            score = float(self._classifier(x))
        flagged = score > self._threshold

        self._verdict = {
            "score": score,
            "flagged": flagged,
            "layer": record.layer,
            "token_pos": record.token_pos,
        }
        signal = ProbeSignal(
            action=ProbeAction.FLAG if flagged else ProbeAction.CONTINUE,
            confidence=score,
            metadata={"score": score, "flagged": flagged, "threshold": self._threshold},
        )
        self._signal_history.append(signal)
        return signal

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        verdict = self._verdict
        if verdict is None:
            # on_activation was never called for this request -- still must
            # return a well-formed ProbeResult, per the Probe contract.
            verdict = {"score": None, "flagged": False, "layer": self._layer, "token_pos": None}

        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict=verdict,
            signal_history=list(self._signal_history),
            metadata={"threshold": self._threshold, "layer": self._layer},
        )

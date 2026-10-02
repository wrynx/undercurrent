"""ActivationCollectorProbe: record activations instead of judging them.

Wire it to an extraction point and every matching activation ends up in the
`ProbeResult`, ready to be stacked into a training set::

    from undercurrent import ProbedModel, ProbeFactory

    spec = {"extraction_points": [{"name": "collect", "layers": 6, "tensor_type": "residual_stream",
                                   "position": "prompt[-1]", "probe_type": "collector",
                                   "probe_kind": "single_shot"}]}
    probes = {"collector": ProbeFactory(ActivationCollectorProbe, {"max_tokens": 1})}
    with ProbedModel.from_pretrained("openai-community/gpt2", spec=spec, probes=probes) as m:
        out = m.generate("Hello there", max_new_tokens=1)
    vector = out.probe_results["collect"].verdict[0]["tensor"]  # torch.Tensor, [hidden]
"""

from __future__ import annotations

from typing import Any

import torch

from undercurrent import ActivationRecord, Probe, ProbeResult, ProbeSignal, RequestContext


class ActivationCollectorProbe(Probe):
    """single_shot probe that keeps a detached CPU copy of each activation it sees.

    Verdict (set in `on_end`): a list with one dict per captured activation,
    in arrival order::

        {"tensor": torch.Tensor, "layer": int, "token_pos": int,
         "tensor_type": str, "is_generated": bool}

    Metadata: `{"captured": int, "dropped": int, "max_tokens": int | None}`.

    Args:
        max_tokens: keep at most this many activations; later ones are
            counted in `dropped` and otherwise ignored. None keeps them all.
    """

    probe_kind = "single_shot"

    def __init__(self, max_tokens: int | None = None) -> None:
        super().__init__()
        if max_tokens is not None and max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1 or None, got {max_tokens}")
        self._max_tokens = max_tokens
        self._captured: list[dict[str, Any]] = []
        self._dropped = 0

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        if self._max_tokens is not None and len(self._captured) >= self._max_tokens:
            self._dropped += 1
            return None
        # clone(): on CPU, .to("cpu") is a no-op view into the adapter's
        # whole [batch, seq, hidden] output; a copy lets that buffer be freed.
        tensor = torch.as_tensor(record.tensor).detach().to("cpu").clone()
        self._captured.append(
            {
                "tensor": tensor,
                "layer": record.layer,
                "token_pos": record.token_pos,
                "tensor_type": record.tensor_type,
                "is_generated": record.is_generated,
            }
        )
        return None

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict=list(self._captured),
            metadata={"captured": len(self._captured), "dropped": self._dropped, "max_tokens": self._max_tokens},
        )

"""A trained probe: a linear (or small MLP) head over one layer's activations.

`train_probe.py` writes a probe directory; this module reads it back and runs
it inline through `ProbedModel`::

    from linear_probe import PROBE_TYPE, load_probe, read_meta, spec_for

    factory = load_probe("outputs/train_probe")
    meta = read_meta("outputs/train_probe")
    with ProbedModel.from_pretrained(meta["base_model"], spec=spec_for(meta),
                                     probes={PROBE_TYPE: factory}) as m:
        out = m.generate("...")  # out.aborted is True when the probe fires

Probe directory format (`FORMAT_VERSION` 1):

- `probe.safetensors`: the `ProbeHead` state dict (`mean`, `std` and the
  `net.*` weights), float32, written with `safetensors.torch.save_file`.
- `probe.json`: `format`, `format_version`, `architecture` (`linear` | `mlp`),
  `input_dim`, `hidden_dim` (mlp only), `layers`, `tensor_type`, `position`,
  `base_model`, `label_names`, `threshold` and `metrics`.

Loading never unpickles anything (only the `safetensors` loader and
`json`), so a probe directory from someone else can't run code on your
machine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from undercurrent import (
    ActivationRecord,
    Probe,
    ProbeAction,
    ProbeFactory,
    ProbeResult,
    ProbeSignal,
    RequestContext,
)

FORMAT = "undercurrent-example-probe"
FORMAT_VERSION = 1
WEIGHTS_FILE = "probe.safetensors"
META_FILE = "probe.json"
ARCHITECTURES = ("linear", "mlp")
PROBE_TYPE = "trained_linear_probe"
REQUIRED_META = (
    "architecture",
    "input_dim",
    "layers",
    "tensor_type",
    "position",
    "base_model",
    "label_names",
    "threshold",
)


class ProbeHead(nn.Module):
    """Standardise the activation, then map it to one logit (positive class).

    `mean`/`std` are buffers fitted on the training set, saved with the
    weights so the probe sees inputs on the scale it was trained on.
    """

    def __init__(self, input_dim: int, architecture: str = "linear", hidden_dim: int = 64) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
        self.input_dim = input_dim
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))
        if architecture == "linear":
            self.net: nn.Module = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std).squeeze(-1)


class TrainedLinearProbe(Probe):
    """single_shot probe that scores activations with a trained `ProbeHead`.

    Each activation gets `score = sigmoid(head(activation))`, the probability
    of `label_names[1]`. A score >= `threshold` returns an ABORT signal (an
    inline extraction point then stops generation); otherwise CONTINUE.

    Verdict: `{"score", "label", "flagged", "layer", "token_pos"}` for the
    highest-scoring activation seen, or None if none matched.

    Build it with `load_probe(dir)`, which binds `head` and the saved
    `threshold`. A spec's `probe_args: {threshold: 0.9}` overrides the
    threshold per extraction point. The head is shared read-only by every
    spawned instance; per-request state lives on `self`.
    """

    probe_kind = "single_shot"

    def __init__(
        self, head: ProbeHead, threshold: float = 0.5, label_names: tuple[str, str] = ("negative", "positive")
    ) -> None:
        super().__init__()
        self._head = head
        self._threshold = float(threshold)
        self._label_names = tuple(label_names)
        self._best: dict[str, Any] | None = None
        self._signals: list[ProbeSignal] = []

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def score(self, tensor: Any) -> float:
        """Probability of the positive label for one activation vector."""
        x = torch.as_tensor(tensor).detach()
        if x.shape[-1] != self._head.input_dim:
            raise ValueError(
                f"activation has {x.shape[-1]} features but the probe was trained on {self._head.input_dim}; "
                "use the base model, layer and tensor_type recorded in probe.json"
            )
        x = x.to(device=self._head.mean.device, dtype=self._head.mean.dtype)
        with torch.no_grad():
            return float(torch.sigmoid(self._head(x)))

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        score = self.score(record.tensor)
        flagged = score >= self._threshold
        if self._best is None or score > self._best["score"]:
            self._best = {
                "score": score,
                "label": self._label_names[int(flagged)],
                "flagged": flagged,
                "layer": record.layer,
                "token_pos": record.token_pos,
            }
        signal = ProbeSignal(
            action=ProbeAction.ABORT if flagged else ProbeAction.CONTINUE,
            confidence=score,
            metadata={"score": score, "threshold": self._threshold, "label": self._label_names[int(flagged)]},
        )
        self._signals.append(signal)
        return signal

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict=self._best,
            signal_history=list(self._signals),
            metadata={"threshold": self._threshold},
        )


def save_probe(out_dir: str | Path, model: nn.Module, meta: Mapping[str, Any]) -> Path:
    """Write `probe.safetensors` and `probe.json` into `out_dir` (created if needed).

    `meta` needs every key in `REQUIRED_META`; for a `ProbeHead`, the
    architecture, input_dim and hidden_dim are filled in from the model.
    Returns `out_dir` as a Path.
    """
    out = Path(out_dir)
    record = dict(meta)
    if isinstance(model, ProbeHead):
        record.setdefault("architecture", model.architecture)
        record.setdefault("input_dim", model.input_dim)
        if model.architecture == "mlp":
            record.setdefault("hidden_dim", model.hidden_dim)
    missing = [key for key in REQUIRED_META if key not in record]
    if missing:
        raise ValueError(f"save_probe: meta is missing {', '.join(missing)}")
    record = {"format": FORMAT, "format_version": FORMAT_VERSION, **record}

    out.mkdir(parents=True, exist_ok=True)
    tensors = {name: t.detach().to("cpu", torch.float32).contiguous() for name, t in model.state_dict().items()}
    save_file(tensors, str(out / WEIGHTS_FILE), metadata={"format": FORMAT})
    (out / META_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out


def read_meta(out_dir: str | Path) -> dict[str, Any]:
    """Read and check `probe.json`: format version, required keys, architecture."""
    path = Path(out_dir) / META_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; train a probe first with examples/train_probe/train_probe.py")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    version = meta.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version={version!r}, but this loader reads version {FORMAT_VERSION}. "
            "Re-train the probe with this version of train_probe.py, or load it with the code that wrote it."
        )
    missing = [key for key in REQUIRED_META if key not in meta]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}; re-train the probe or restore the file")
    if meta["architecture"] not in ARCHITECTURES:
        raise ValueError(f"{path}: architecture must be one of {ARCHITECTURES}, got {meta['architecture']!r}")
    return meta


def load_head(out_dir: str | Path) -> tuple[ProbeHead, dict[str, Any]]:
    """Rebuild the `ProbeHead` from a probe directory. Returns `(head, meta)`."""
    meta = read_meta(out_dir)
    weights_path = Path(out_dir) / WEIGHTS_FILE
    if not weights_path.is_file():
        raise FileNotFoundError(f"{weights_path} not found; probe.json has no weights next to it")
    state = load_file(str(weights_path))

    input_dim = int(meta["input_dim"])
    first_weight = state.get("net.weight", state.get("net.0.weight"))
    actual = None if first_weight is None else first_weight.shape[-1]
    if actual != input_dim or state.get("mean", torch.empty(0)).shape != (input_dim,):
        raise ValueError(
            f"{Path(out_dir) / META_FILE} says input_dim={input_dim}, but {weights_path.name} holds weights for "
            f"{actual} input features. The two files don't belong together: re-run train_probe.py, or restore "
            "the probe.json that was written with these weights."
        )
    head = ProbeHead(input_dim, meta["architecture"], int(meta.get("hidden_dim", 64)))
    try:
        head.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"{weights_path} doesn't match a {meta['architecture']!r} probe as described in probe.json: {exc}"
        ) from exc
    head.eval()
    return head, meta


def load_probe(out_dir: str | Path) -> ProbeFactory:
    """Load a probe directory into a `ProbeFactory` for `ProbedModel(probes={...})`."""
    head, meta = load_head(out_dir)
    return ProbeFactory(
        TrainedLinearProbe,
        {"head": head, "threshold": float(meta["threshold"]), "label_names": tuple(meta["label_names"])},
    )


def spec_for(meta: Mapping[str, Any], name: str = "trained_probe") -> dict[str, Any]:
    """An inline extraction point at the layer, tensor type and position the probe was trained on."""
    return {
        "extraction_points": [
            {
                "name": name,
                "layers": list(meta["layers"]),
                "tensor_type": meta["tensor_type"],
                "position": meta["position"],
                "probe_type": PROBE_TYPE,
                "probe_kind": "single_shot",
            }
        ]
    }

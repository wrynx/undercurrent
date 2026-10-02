"""CustomMLPProbe: a single_shot probe wrapping a REAL, externally-trained
`SafetyClassifierHead` checkpoint loaded from disk -- as opposed to
`SingleTokenSafetyProbe` (`single_token.py`), which uses a random-initialized
one to validate platform wiring only, never real model quality.

Checkpoint format
------------------
`torch.save(model.state_dict(), path)` for a `SafetyClassifierHead(input_dim,
hidden_sizes)` (see `models.py`) -- a `Linear -> ReLU -> [Dropout ->]`
stack repeated per hidden size, ending in a final `Linear` to either a
1-dim sigmoid-scored logit or an N-way softmax head, all named `net`.
`input_dim`, `hidden_sizes`, `output_dim`, and whether `Dropout` modules
are present are all inferred from the state dict's own tensor shapes and
`net.<i>.weight` key layout (see `_infer_shape_from_state_dict`), so no
separate metadata file is needed. A `{"state_dict": {...}}` wrapper dict is
also tolerated, for checkpoints saved alongside other training metadata.
Checkpoints are loaded with `torch.load(..., weights_only=True)`: only plain
tensor state_dicts (optionally inside that wrapper) are accepted, and a file
containing any other pickled object is rejected with `UnsafeCheckpointError`
rather than unpickled -- a crafted checkpoint must never run code. Loading
requires torch>=2.6, the first release whose `weights_only` loader is not
bypassable (CVE-2025-32434).
See `examples/make_dummy_mlp_checkpoint.py` for a script that produces a
placeholder checkpoint in this format (random weights -- for wiring
smoke-tests, not a real classifier), and
`examples/serve_llama_mlp_pipeline.py` for this probe attached to a live
Llama model.

Checkpoint caching
-------------------
`Probe` instances are spawned fresh per request (see `undercurrent.core`'s
isolation model) -- re-reading a checkpoint from disk and reconstructing
the module on every single request would be wasteful, and actively
counterproductive under concurrent serving load (many requests spawning
probes at once). `_load_cached_classifier` loads each distinct
`(model_path, device)` exactly once into a module-level cache and hands
every spawned probe instance a reference to the SAME loaded module.
Sharing a module across concurrent requests this way is safe here
specifically because it's used in eval() mode, called only inside
`torch.no_grad()`, and never mutated in place after loading -- a stateless,
read-only forward pass has no cross-request state to race on.
"""

from __future__ import annotations

import logging
import pickle
import re
import threading
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

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()
_classifier_cache: dict[tuple[str, str, str, bool], SafetyClassifierHead] = {}


# First torch release whose `weights_only=True` loader is hardened against
# CVE-2025-32434 (arbitrary code execution despite `weights_only=True`).
_MIN_SAFE_TORCH = (2, 6)


class UnsafeCheckpointError(ValueError):
    """Raised when a checkpoint can't be loaded as a plain tensor state_dict
    under `torch.load(..., weights_only=True)` -- i.e. it contains pickled
    objects other than tensors and plain containers. Never fall back to full
    unpickling: that would execute arbitrary code from the file."""


def _torch_version_tuple() -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _load_state_dict_weights_only(model_path: str, device: str) -> dict[str, Any]:
    """Load `model_path` as a tensor state_dict, never unpickling arbitrary
    objects. Accepts a bare state_dict or a `{"state_dict": {...}}` wrapper."""
    if _torch_version_tuple() < _MIN_SAFE_TORCH:
        raise RuntimeError(
            f"CustomMLPProbe needs torch>={_MIN_SAFE_TORCH[0]}.{_MIN_SAFE_TORCH[1]} to load checkpoints "
            f"safely (installed: {torch.__version__}); older releases' weights_only loader can be "
            "bypassed by a crafted file (CVE-2025-32434). Upgrade torch."
        )
    try:
        raw = torch.load(model_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise UnsafeCheckpointError(
            f"Refusing to load checkpoint {model_path!r}: it contains objects other than tensors, so it "
            "can't be loaded with torch.load(weights_only=True). Only plain tensor state_dicts (or "
            "{'state_dict': state_dict}) are accepted. Re-save it from trusted code with "
            "`torch.save(model.state_dict(), path)`."
        ) from exc
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(state_dict, dict) or not all(isinstance(v, torch.Tensor) for v in state_dict.values()):
        raise UnsafeCheckpointError(
            f"Checkpoint {model_path!r} is not a tensor state_dict. Re-save it with "
            "`torch.save(model.state_dict(), path)`."
        )
    return state_dict


_NET_WEIGHT_RE = re.compile(r"^net\.(\d+)\.weight$")


def _infer_shape_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int, list[int], float, int]:
    """Recover `(input_dim, hidden_sizes, dropout, output_dim)` for
    `SafetyClassifierHead` from a checkpoint's own tensor shapes and `net.*`
    key layout -- so that both the original single-hidden-layer stub and a
    deeper, dropout-using externally-trained checkpoint round-trip through
    the same inference path. The gap between consecutive `Linear` indices in
    `net` (2 for `Linear -> ReLU -> Linear`, 3 for `Linear -> ReLU -> Dropout
    -> Linear`) tells us whether `Dropout` modules need to be reinserted for
    `load_state_dict` to line up; the exact dropout probability doesn't
    matter for inference, since `model.eval()` disables it regardless.
    `output_dim` is the final `Linear`'s output width -- 1 for a
    sigmoid-scored checkpoint, >1 for one trained with an N-way softmax head
    (see `SafetyClassifierHead.forward`).
    """
    indices = sorted(int(match.group(1)) for key in state_dict if (match := _NET_WEIGHT_RE.match(key)))
    if not indices:
        raise ValueError(
            "checkpoint state_dict is missing any 'net.<i>.weight' keys -- CustomMLPProbe expects a "
            "SafetyClassifierHead-shaped checkpoint: torch.save(model.state_dict(), path) for a "
            "Linear -> ReLU -> [Dropout ->] ... -> Linear head named 'net' (see "
            "models.SafetyClassifierHead and this module's docstring)."
        )
    weights = [state_dict[f"net.{i}.weight"] for i in indices]
    input_dim = int(weights[0].shape[1])
    hidden_sizes = [int(w.shape[0]) for w in weights[:-1]]
    output_dim = int(weights[-1].shape[0])
    dropout = 0.1 if len(indices) >= 2 and indices[1] - indices[0] == 3 else 0.0
    return input_dim, hidden_sizes, dropout, output_dim


def _load_cached_classifier(
    model_path: str,
    device: str,
    activation: str = "relu",
    final_activation: bool = False,
) -> SafetyClassifierHead:
    """Load (or return an already-loaded) `SafetyClassifierHead` from
    `model_path`, cached at module scope keyed by `(model_path, device,
    activation, final_activation)` -- see this module's docstring for why
    sharing one loaded instance across every spawned probe/request is safe.
    `activation`/`final_activation` can't be recovered from the checkpoint's
    own tensor shapes (see `SafetyClassifierHead`'s docstring), so they must
    match how the checkpoint was actually trained or `load_state_dict` will
    succeed while silently producing wrong scores. A double-checked lock
    avoids two concurrent first-time loads of the same checkpoint racing
    (harmless if it happened -- just wasteful) without serializing every
    subsequent cache hit behind a lock.
    """
    key = (model_path, device, activation, final_activation)
    cached = _classifier_cache.get(key)
    if cached is not None:
        return cached
    with _cache_lock:
        cached = _classifier_cache.get(key)
        if cached is not None:
            return cached
        state_dict = _load_state_dict_weights_only(model_path, device)
        input_dim, hidden_sizes, dropout, output_dim = _infer_shape_from_state_dict(state_dict)
        model = SafetyClassifierHead(
            input_dim,
            hidden_sizes,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
            final_activation=final_activation,
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        _classifier_cache[key] = model
        logger.info(
            "content_safety_demo.custom_mlp: loaded checkpoint from %r "
            "(input_dim=%d, hidden_sizes=%r, output_dim=%d, device=%r)",
            model_path,
            input_dim,
            hidden_sizes,
            output_dim,
            device,
        )
        return model


@register_probe("custom_mlp_safety_check")
class CustomMLPProbe(Probe):
    """single_shot probe: one activation in, one safety verdict out, via a
    REAL trained `SafetyClassifierHead` checkpoint loaded from `model_path`.

    Verdict shape (identical to `SingleTokenSafetyProbe`'s, so callers can
    swap between the two without touching downstream code):
        {"score": Optional[float], "flagged": bool, "layer": int, "token_pos": Optional[int]}
    """

    probe_kind = "single_shot"

    def __init__(
        self,
        layer: int,
        model_path: str,
        threshold: float = 0.5,
        device: str = "cpu",
        activation: str = "relu",
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        self._layer = layer
        self._model_path = model_path
        self._threshold = threshold
        self._device = device
        self._classifier = _load_cached_classifier(
            model_path, device, activation=activation, final_activation=final_activation
        )

        self._verdict: dict[str, Any] | None = None
        self._signal_history: list[ProbeSignal] = []
        self._called = False

    def on_start(self, request_ctx: RequestContext) -> None:
        pass  # nothing to set up beyond what __init__ already initialized

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

        x = to_flat_tensor(record.tensor).to(self._device)
        expected_dim = self._classifier.net[0].in_features
        if x.shape[0] != expected_dim:
            raise ValueError(
                f"{type(self).__name__}: checkpoint {self._model_path!r} expects an activation of "
                f"dim {expected_dim}, got {x.shape[0]} -- wrong checkpoint for this model, or the "
                "extraction point's layer/tensor_type doesn't match what the checkpoint was trained on"
            )

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
            metadata={"threshold": self._threshold, "layer": self._layer, "model_path": self._model_path},
        )

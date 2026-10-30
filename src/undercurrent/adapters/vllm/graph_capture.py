"""Graph-mode residual-stream capture: probing without `enforce_eager=True`.

Forward hooks are Python, and vLLM's CUDA-graph replay (and its
`torch.compile`d forward) never runs them, so hook-based capture needs eager
execution. On an A100 that costs about 2x decode throughput for a 7B model and
about 8x for a 0.5B one, before any probe runs.

vLLM already has a graph-compatible way to read residual-stream activations:
the auxiliary hidden-state outputs that EAGLE-3 speculative decoding uses.
Models that implement it (Llama and Qwen2 among them, via
`EagleModelMixin._maybe_add_hidden_state`) append `hidden_states + residual`
after every decoder layer listed in `aux_hidden_state_layers` to an extra
output list, inside the compiled and captured graph. Aux index `k + 1` is the
output of decoder layer `k`: the same tensor this adapter's `residual_stream`
captures at layer `k`.

How it is wired:

1. `VLLMEngineAdapter.load_model(graph_capture_layers=[...])` sets
   `UNDERCURRENT_VLLM_GRAPH_CAPTURE_LAYERS` (inherited by vLLM's engine
   process) and disables vLLM's compile cache, whose key doesn't include the
   aux layers we add.
2. In the engine process, the `vllm.general_plugins` entry point
   (`plugin.py`) sees the variable and patches `GPUModelRunner.load_model`
   (`install_runner_patch`). After the model loads, and before vLLM compiles
   it and captures CUDA graphs, the patch calls
   `set_aux_hidden_state_layers` and sets `use_aux_hidden_state_outputs`,
   exactly as vLLM does for EAGLE-3, and wraps `_model_forward` so the aux
   outputs reach a callback (`AUX_CALLBACK_ATTR`) after each forward pass.
3. The worker extension installs no forward hooks in this mode. Its callback
   stages each probed layer's matched rows from the aux outputs, with the same
   batched, non-blocking device-to-host copy as the hook path.

Limits: `residual_stream` only, on the layers fixed at load time; models must
implement vLLM's EAGLE-3 aux-output interface; incompatible with speculative
decoding methods that use aux outputs themselves. Each limit fails loudly.

Internal and experimental: this reads undocumented vLLM internals and is not
part of the public API (see docs/api-stability.md).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from ...errors import ProbingRuntimeError
from ...spec import ExtractionPoint, TensorType

logger = logging.getLogger(__name__)

GRAPH_CAPTURE_ENV = "UNDERCURRENT_VLLM_GRAPH_CAPTURE_LAYERS"
# Set on the GPUModelRunner instance by the patched load_model: the decoder layers whose
# residual stream the model now outputs, in aux-output order. Its presence means graph mode.
LAYERS_ATTR = "_undercurrent_graph_capture_layers"
# Set on the GPUModelRunner instance by the worker extension: called with the aux outputs.
AUX_CALLBACK_ATTR = "_undercurrent_on_aux_hidden_states"
_PATCHED_ATTR = "_undercurrent_graph_capture_patched"


class GraphCaptureError(ProbingRuntimeError):
    """The loaded model or engine configuration can't support graph-mode capture."""


def encode_layers(layers: Iterable[int]) -> str:
    return ",".join(str(layer) for layer in sorted({int(layer) for layer in layers}))


def decode_layers(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(part) for part in value.split(",") if part.strip()}))


def unsupported_extraction_points(points: Iterable[ExtractionPoint], layers: Sequence[int]) -> list[str]:
    """Why each of `points` can't be captured in graph mode (empty if all can)."""
    allowed = set(layers)
    problems = []
    for ep in points:
        if TensorType(ep.tensor_type) != TensorType.RESIDUAL_STREAM:
            problems.append(f"{ep.name!r}: tensor_type={TensorType(ep.tensor_type).value!r} (only residual_stream)")
            continue
        missing = sorted(set(ep.layers) - allowed)
        if missing:
            problems.append(f"{ep.name!r}: layers {missing} are not in graph_capture_layers {sorted(allowed)}")
    return problems


def configure_runner(runner: Any, layers: Sequence[int]) -> None:
    """Make a loaded GPUModelRunner output the residual stream at `layers` (call before compilation)."""
    layers = tuple(sorted(set(layers)))
    if layers:
        model = runner.get_model()
        if not hasattr(model, "set_aux_hidden_state_layers"):
            raise GraphCaptureError(
                f"{type(model).__name__} doesn't implement vLLM's EAGLE-3 aux hidden-state outputs, which "
                "graph-mode capture uses. Load it with enforce_eager=True and without graph_capture_layers."
            )
        if getattr(runner, "use_aux_hidden_state_outputs", False):
            raise GraphCaptureError(
                "this engine's speculative decoding already uses the model's aux hidden-state outputs; "
                "graph-mode capture can't share them. Disable speculative decoding, or use enforce_eager=True "
                "without graph_capture_layers."
            )
        model.set_aux_hidden_state_layers(tuple(layer + 1 for layer in layers))
        runner.use_aux_hidden_state_outputs = True
    setattr(runner, LAYERS_ATTR, layers)

    original_forward = runner._model_forward

    def _model_forward(*args: Any, **kwargs: Any) -> Any:
        output = original_forward(*args, **kwargs)
        callback = getattr(runner, AUX_CALLBACK_ATTR, None)
        if callback is not None and layers and isinstance(output, tuple):
            callback(output[1])
        return output

    runner._model_forward = _model_forward
    logger.info("undercurrent.adapters.vllm: graph-mode capture of residual_stream at layers %s", list(layers))


def install_runner_patch(runner_cls: Any, layers: Sequence[int]) -> None:
    """Wrap `runner_cls.load_model` so every runner is configured right after loading. Idempotent."""
    if getattr(runner_cls, _PATCHED_ATTR, False):
        return
    original_load_model = runner_cls.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        configure_runner(self, layers)
        return result

    runner_cls.load_model = load_model
    setattr(runner_cls, _PATCHED_ATTR, True)

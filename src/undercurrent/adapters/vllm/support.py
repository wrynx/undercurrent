"""Static, read-only description of what the vLLM adapter can capture.

Used by ``undercurrent inspect-model`` to answer "what could I probe on
vLLM?" without importing vllm (which may not be installed) or loading a
model. It mirrors `worker_extension.ProbingWorkerExtension`: keep the two in
step when the worker's hook points change (``tests/cli`` checks they agree).

The attribute paths describe vLLM's own model implementations, which use the
same module names as their transformers counterparts for the common
families (Llama, Qwen, Mistral, GPT-2, ...). Applying them to a transformers
model is therefore a close proxy, not a guarantee.

Internal: not part of the public API (see docs/api-stability.md) and may
change in any release.
"""

from __future__ import annotations

from typing import Any

from ...spec import TensorType

#: tensor types the vLLM adapter can't capture, with the reason.
_UNSUPPORTED_REASONS: dict[TensorType, str] = {
    TensorType.KV: (
        "the KV cache lives in paged blocks, not a per-layer forward-pass tensor; the vLLM adapter doesn't capture it"
    ),
}

#: Where `ProbingWorkerExtension` looks for each hook point, relative to the
#: model (attribute paths, first match wins).
DECODER_LAYER_PATHS: tuple[str, ...] = ("model.layers", "layers", "transformer.h")
ATTENTION_ATTR = "self_attn"
MLP_ATTR = "mlp"
FINAL_NORM_PATHS: tuple[str, ...] = ("model.norm", "norm", "transformer.ln_f")


def supported_tensor_types() -> dict[TensorType, str | None]:
    """Every `TensorType` mapped to ``None`` if the vLLM adapter can
    capture it, or to a short reason why it can't."""
    return {tensor_type: _UNSUPPORTED_REASONS.get(tensor_type) for tensor_type in TensorType}


def resolve_path(model: Any, path: str) -> Any:
    """Follow a dotted attribute path (``"model.layers"``), or return None."""
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def model_tensor_support(model: Any) -> dict[TensorType, str | None]:
    """Like `supported_tensor_types()`, narrowed to one model's
    structure (found via the attribute paths above). ``model`` may live on
    the ``meta`` device; nothing is run."""
    support = supported_tensor_types()
    layers = None
    for path in DECODER_LAYER_PATHS:
        candidate = resolve_path(model, path)
        if candidate is not None and len(candidate) > 0:
            layers = candidate
            break
    if layers is None:
        reason = f"no decoder-layer list at any of {', '.join(DECODER_LAYER_PATHS)}"
        return {t: support[t] or reason for t in support}

    first = layers[0]
    if support[TensorType.ATTN_OUT] is None and getattr(first, ATTENTION_ATTR, None) is None:
        support[TensorType.ATTN_OUT] = (
            f"decoder layers have no '{ATTENTION_ATTR}' submodule for the vLLM adapter to hook"
        )
    if support[TensorType.MLP_OUT] is None and getattr(first, MLP_ATTR, None) is None:
        support[TensorType.MLP_OUT] = f"decoder layers have no '{MLP_ATTR}' submodule for the vLLM adapter to hook"
    if support[TensorType.FINAL_NORM] is None and all(resolve_path(model, p) is None for p in FINAL_NORM_PATHS):
        support[TensorType.FINAL_NORM] = f"no final norm at any of {', '.join(FINAL_NORM_PATHS)}"
    return support


__all__ = [
    "ATTENTION_ATTR",
    "DECODER_LAYER_PATHS",
    "FINAL_NORM_PATHS",
    "MLP_ATTR",
    "model_tensor_support",
    "resolve_path",
    "supported_tensor_types",
]

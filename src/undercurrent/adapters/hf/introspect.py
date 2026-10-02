"""Model-structure helpers shared by `HFEngineAdapter` and `undercurrent inspect-model`.

These answer "where would the HF adapter hook this model?" without running
it, so they work on a model instantiated on the ``meta`` device (no weights
allocated)::

    from undercurrent.adapters.hf.introspect import find_decoder_layers, supported_tensor_types

    layers = find_decoder_layers(model)          # {0: GPT2Block, 1: GPT2Block, ...}
    supported_tensor_types()[TensorType.KV]      # "the KV cache isn't a per-layer ..." (a reason)

`HFEngineAdapter` installs its hooks with exactly these functions, so what
the CLI reports is what the adapter does.

Internal: not part of the public API (see docs/api-stability.md) and may
change in any release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...spec import TensorType
from .errors import HFAdapterLimitationError

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime torch dependency
    from torch import nn

#: tensor types the HF adapter can't capture, with the reason. See
#: adapter.py's module docstring.
_UNSUPPORTED_REASONS: dict[TensorType, str] = {
    TensorType.KV: (
        "the KV cache isn't a per-layer forward-pass tensor with one row per token; the HF adapter doesn't capture it"
    ),
    TensorType.FINAL_NORM: "the HF adapter doesn't hook the model-level final norm (the vLLM adapter does)",
}


def supported_tensor_types() -> dict[TensorType, str | None]:
    """Every `TensorType` mapped to ``None`` if the HF adapter can
    capture it, or to a short reason why it can't."""
    return {tensor_type: _UNSUPPORTED_REASONS.get(tensor_type) for tensor_type in TensorType}


def find_decoder_layers(model: Any) -> dict[int, nn.Module]:
    """Locate the model's indexable decoder-layer list, as ``{index: layer}``.

    Checks ``model.model.layers`` / ``model.layers`` (Llama family) and
    ``model.transformer.h`` / ``model.h`` (GPT-2 style). Raises
    `HFAdapterLimitationError` for any other architecture.

    See `undercurrent.adapters.vllm.worker_extension.ProbingWorkerExtension._find_decoder_layers`
    for the vLLM equivalent (independently maintained: each adapter's exact
    hook point differs enough that a shared helper would need
    engine-specific branches anyway).
    """
    inner = getattr(model, "model", None) or getattr(model, "transformer", None) or model
    candidates = [
        getattr(inner, "layers", None),  # Llama-family
        getattr(inner, "h", None),  # GPT-2-style
    ]
    for candidate in candidates:
        if candidate is not None and len(candidate) > 0:
            return dict(enumerate(candidate))
    raise HFAdapterLimitationError(
        f"could not locate a decoder-layer list on {type(model).__name__} (checked "
        "model.model.layers/model.layers and model.transformer.h/model.h). This "
        "model's architecture isn't supported by undercurrent.adapters.hf yet. Run "
        "`undercurrent inspect-model MODEL` to see what can be hooked, and please report the model at "
        "https://github.com/wrynx/undercurrent/issues (the fix is adding its layer-list attribute path "
        "to undercurrent.adapters.hf.introspect.find_decoder_layers())."
    )


def find_attention(layer: Any) -> nn.Module | None:
    """The decoder layer's attention submodule (hooked for ``attn_out``), or None."""
    return getattr(layer, "self_attn", None) or getattr(layer, "attn", None)


def find_mlp(layer: Any) -> nn.Module | None:
    """The decoder layer's MLP submodule (hooked for ``mlp_out``), or None."""
    return getattr(layer, "mlp", None)


def model_tensor_support(model: Any) -> dict[TensorType, str | None]:
    """Like `supported_tensor_types()`, narrowed to one model's
    structure: ``attn_out`` / ``mlp_out`` need the submodules the adapter
    hooks. ``model`` may live on the ``meta`` device; nothing is run.
    Raises `HFAdapterLimitationError` if the decoder layers can't be
    found (see `find_decoder_layers()`)."""
    support = supported_tensor_types()
    first = find_decoder_layers(model)[0]
    if support[TensorType.ATTN_OUT] is None and find_attention(first) is None:
        support[TensorType.ATTN_OUT] = "decoder layers have no 'self_attn' or 'attn' submodule to hook"
    if support[TensorType.MLP_OUT] is None and find_mlp(first) is None:
        support[TensorType.MLP_OUT] = "decoder layers have no 'mlp' submodule to hook"
    return support


__all__ = [
    "HFAdapterLimitationError",
    "find_attention",
    "find_decoder_layers",
    "find_mlp",
    "model_tensor_support",
    "supported_tensor_types",
]

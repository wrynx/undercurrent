"""`residual_stream` capture on vLLM's two decoder-layer contracts, with real
torch on CPU (no vLLM engine, no GPU).

vLLM's fused-residual decoder layers (Llama, Qwen2, Gemma 2, ...) return
`(hidden_states, residual)` and leave the add to the next layer's fused
`input_layernorm`; the residual stream after the layer is their sum.
GPT-2-style layers add the residual themselves and return one tensor. See
`_residual_stream_from_layer_output()` in
`undercurrent/adapters/vllm/worker_extension.py`.

The stacks below copy vLLM 0.28's `LlamaDecoderLayer.forward` /
`LlamaModel.forward` / `GPT2Block.forward` control flow, and the fused norm
mutates both of its inputs in place the way vLLM's CUDA `fused_add_rms_norm`
kernel does, so a capture that aliases the layer output gets corrupted when
the next layer runs. The expected values come from a separate pure-torch
residual-stream computation, not from the layers' own outputs.

`test_upstream_*` read the installed vLLM's source (skipped without vLLM) so a
change to the contract trips CI instead of silently breaking capture.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import types

import pytest
import torch
from torch import nn

from tests.adapters.vllm._helpers import make_extraction_point
from undercurrent.adapters.vllm import VLLMAdapterLimitationError
from undercurrent.adapters.vllm.worker_extension import (
    ProbingWorkerExtension,
    _extract_captured_tensor,
    _residual_stream_from_layer_output,
)
from undercurrent.core import RequestContext
from undercurrent.spec import TensorType

HIDDEN = 8
NUM_LAYERS = 3
NUM_TOKENS = 4
EPS = 1e-6


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * weight


class FusedAddRMSNorm(nn.Module):
    """vLLM's `RMSNorm`: `norm(x)`, or `norm(x, residual)` -> `(norm(x + residual), x + residual)`.
    The fused form writes both results into its inputs, like the CUDA kernel."""

    inplace = True

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(1.0 + 0.1 * torch.randn(HIDDEN))

    def forward(self, x, residual=None):
        if residual is None:
            return _rms_norm(x, self.weight)
        if not self.inplace:
            residual = x + residual
            return _rms_norm(residual, self.weight), residual
        residual.add_(x)
        x.copy_(_rms_norm(residual, self.weight))
        return x, residual


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(HIDDEN, 2 * HIDDEN)
        self.down = nn.Linear(2 * HIDDEN, HIDDEN)

    def forward(self, x):
        return self.down(torch.nn.functional.gelu(self.up(x)))


class ToyAttention(nn.Module):
    """Stands in for `self_attn`; a per-token linear map is enough here."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, positions, hidden_states):
        return self.proj(hidden_states)


class FusedResidualLayer(nn.Module):
    """vLLM 0.28's `LlamaDecoderLayer.forward`, line for line."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = FusedAddRMSNorm()
        self.post_attention_layernorm = FusedAddRMSNorm()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SingleTensorLayer(nn.Module):
    """vLLM's `GPT2Block.forward`: adds the residual itself, returns one tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.ln_1 = FusedAddRMSNorm()
        self.ln_2 = FusedAddRMSNorm()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, positions, hidden_states):
        residual = hidden_states
        hidden_states = residual + self.self_attn(positions, self.ln_1(hidden_states))
        residual = hidden_states
        return residual + self.mlp(self.ln_2(hidden_states))


class ToyModel(nn.Module):
    """`model.model.layers` / `model.model.norm`, run like vLLM's `LlamaModel.forward`."""

    def __init__(self, fused: bool) -> None:
        super().__init__()
        self.fused = fused
        layer_cls = FusedResidualLayer if fused else SingleTensorLayer
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layer_cls() for _ in range(NUM_LAYERS))
        self.model.norm = FusedAddRMSNorm()

    @torch.inference_mode()
    def forward(self, positions, embeddings):
        hidden_states = embeddings.clone()
        if not self.fused:
            for layer in self.model.layers:
                hidden_states = layer(positions, hidden_states)
            return self.model.norm(hidden_states)
        residual = None
        for layer in self.model.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.model.norm(hidden_states, residual)
        return hidden_states

    @torch.inference_mode()
    def reference_residual_stream(self, embeddings):
        """The residual stream after each layer, computed directly (no fused
        norms, no in-place ops): the quantity HF's decoder-layer output holds."""
        stream = embeddings.clone()
        streams = []
        for layer in self.model.layers:
            norm_1, norm_2 = (
                (layer.input_layernorm, layer.post_attention_layernorm) if self.fused else (layer.ln_1, layer.ln_2)
            )
            stream = stream + layer.self_attn(None, _rms_norm(stream, norm_1.weight))
            stream = stream + layer.mlp(_rms_norm(stream, norm_2.weight))
            streams.append(stream)
        return streams


class _InputBatch:
    def __init__(self, req_ids, num_computed_tokens_cpu):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = num_computed_tokens_cpu


class TorchModelRunner:
    """A model runner whose `execute_model` runs a real forward pass, so the
    worker extension's real `register_forward_hook` hooks fire mid-forward."""

    def __init__(self, model: ToyModel, embeddings: torch.Tensor) -> None:
        self.model = model
        self._embeddings = embeddings
        self.input_batch = None

        def _execute_model(scheduler_output, *a, **kw):
            return self.model(torch.arange(len(self._embeddings)), self._embeddings)

        self.execute_model = _execute_model


def _capture(router, model: ToyModel, embeddings: torch.Tensor, tensor_type=TensorType.RESIDUAL_STREAM):
    """Run one step of `NUM_TOKENS` rows through the worker extension with a
    recording probe on every layer; return `{layer: [rows x HIDDEN]}`."""
    ext = ProbingWorkerExtension()
    runner = TorchModelRunner(model, embeddings)
    ext.model_runner = runner  # type: ignore[attr-defined]
    ext.bind_router(router)
    ext._ensure_hooks_installed()

    ep = make_extraction_point(name="ep", layer=list(range(NUM_LAYERS)), tensor=tensor_type, position="generated[*]")
    ext.register_extraction("req", [ep], prompt_len=1)
    router.register_request(
        "req", [ep], RequestContext(request_id="req", prompt_metadata={}, extraction_point_config=None)
    )
    # One request, NUM_TOKENS rows starting at token 1: all past the 1-token prompt.
    runner.input_batch = _InputBatch(req_ids=["req"], num_computed_tokens_cpu=[1])
    runner.execute_model(types.SimpleNamespace(num_scheduled_tokens={"req": NUM_TOKENS}))

    records = router.get_probe("req", "ep").received
    router.end_request("req")
    by_layer: dict[int, list] = {}
    for record in sorted(records, key=lambda r: (r.layer, r.token_pos)):
        by_layer.setdefault(record.layer, []).append(record.tensor)
    return {layer: torch.stack(rows) for layer, rows in by_layer.items()}


@pytest.fixture
def embeddings():
    torch.manual_seed(0)
    return torch.randn(NUM_TOKENS, HIDDEN)


@pytest.mark.parametrize("fused", [True, False], ids=["fused-residual", "single-tensor"])
def test_residual_stream_matches_reference_across_a_three_layer_stack(router, embeddings, fused):
    torch.manual_seed(1)
    model = ToyModel(fused=fused)
    expected = model.reference_residual_stream(embeddings)

    captured = _capture(router, model, embeddings)

    assert sorted(captured) == list(range(NUM_LAYERS))
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(captured[layer], expected[layer], msg=lambda m, layer=layer: f"layer {layer}: {m}")


def test_mlp_out_is_still_the_mlp_output_on_fused_residual_layers(router, embeddings):
    """Only residual_stream changed: mlp_out still hooks the MLP submodule.

    In-place fused norms on purpose: the next layer's fused add overwrites the
    MLP output tensor, so this also guards `_emit()` copying (not viewing) the
    captured row on CPU."""
    torch.manual_seed(1)
    model = ToyModel(fused=True)
    mlp_outputs = {}
    for idx, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_hook(lambda m, i, out, idx=idx: mlp_outputs.__setitem__(idx, out.clone()))

    captured = _capture(router, model, embeddings, tensor_type=TensorType.MLP_OUT)

    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(captured[layer], mlp_outputs[layer])


def test_fused_residual_capture_is_a_copy():
    hidden_states, residual = torch.randn(NUM_TOKENS, HIDDEN), torch.randn(NUM_TOKENS, HIDDEN)
    expected = hidden_states + residual

    captured = _residual_stream_from_layer_output((hidden_states, residual), layer_idx=0)
    # What the next layer's fused add + norm does to its inputs.
    residual.add_(hidden_states)
    hidden_states.zero_()

    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_fused_residual_sum_matches_vllm_rounding_in_bf16():
    """vLLM's reference fused add computes in float32 and rounds back."""
    torch.manual_seed(2)
    hidden_states = torch.randn(64, HIDDEN).to(torch.bfloat16)
    residual = (100 * torch.randn(64, HIDDEN)).to(torch.bfloat16)

    captured = _residual_stream_from_layer_output((hidden_states, residual), layer_idx=0)

    assert captured.dtype == torch.bfloat16
    vllm_reference = (hidden_states.to(torch.float32) + residual.to(torch.float32)).to(torch.bfloat16)
    assert torch.equal(captured, vllm_reference)


def test_single_tensor_and_already_added_outputs_are_captured_as_is():
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    assert _residual_stream_from_layer_output(hidden_states, layer_idx=0) is hidden_states
    # (hidden_states, None): MiniCPM / FlexOlmo / Gemma 4 add the residual themselves.
    assert _residual_stream_from_layer_output((hidden_states, None), layer_idx=0) is hidden_states


@pytest.mark.parametrize(
    "output",
    [
        pytest.param((torch.randn(4, 8), torch.randn(4, 8), torch.randn(2, 8)), id="3-tuple"),
        pytest.param((torch.randn(4, 8), torch.randn(4, 16)), id="shape-mismatch"),
        pytest.param((torch.randn(4, 8), torch.randn(4, 8).to(torch.float16)), id="dtype-mismatch"),
        pytest.param((torch.randn(4, 8), "aux"), id="non-tensor-second"),
        pytest.param({"hidden_states": torch.randn(4, 8)}, id="dict"),
    ],
)
def test_unrecognised_layer_outputs_fail_loudly(output):
    with pytest.raises(VLLMAdapterLimitationError, match="residual_stream at layer=3"):
        _residual_stream_from_layer_output(output, layer_idx=3)


def test_other_tensor_types_still_take_the_first_tensor():
    first, second = torch.randn(4, 8), torch.randn(4, 8)
    for tensor_type in (TensorType.ATTN_OUT, TensorType.MLP_OUT, TensorType.FINAL_NORM):
        assert _extract_captured_tensor((first, second), tensor_type, layer_idx=0) is first


# ---------------------------------------------------------------------------
# Upstream contract: the installed vLLM's source, read with `ast` (importing
# vLLM's model modules costs ~1 GB and >10 s).
# ---------------------------------------------------------------------------


def _vllm_source(relative: str) -> ast.Module:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("vllm isn't installed")
    path = pathlib.Path(next(iter(spec.submodule_search_locations))) / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _method(tree: ast.Module, class_name: str, method: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return item
    raise AssertionError(f"{class_name}.{method} not found in the installed vLLM")


def _returns(func: ast.FunctionDef) -> list[str]:
    return [ast.unparse(node.value) for node in ast.walk(func) if isinstance(node, ast.Return) and node.value]


def test_upstream_llama_decoder_layer_returns_hidden_states_and_residual():
    forward = _method(_vllm_source("model_executor/models/llama.py"), "LlamaDecoderLayer", "forward")
    assert "residual" in [arg.arg for arg in forward.args.args]
    assert _returns(forward) == ["(hidden_states, residual)"]


def test_upstream_vllm_sums_hidden_states_and_residual_for_the_stream():
    """vLLM's own residual-stream materialisation (EAGLE aux hidden states)
    uses the same `hidden_states + residual` convention."""
    method = _method(_vllm_source("model_executor/models/interfaces.py"), "EagleModelMixin", "_maybe_add_hidden_state")
    assert "hidden_states + residual" in ast.unparse(method)


def test_upstream_gpt2_block_returns_a_single_tensor():
    forward = _method(_vllm_source("model_executor/models/gpt2.py"), "GPT2Block", "forward")
    assert [arg.arg for arg in forward.args.args] == ["self", "hidden_states"]
    assert _returns(forward) == ["hidden_states"]

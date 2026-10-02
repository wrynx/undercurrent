"""undercurrent.adapters.vllm.support must agree with ProbingWorkerExtension's hook points."""

from types import SimpleNamespace

import pytest

from undercurrent.adapters.vllm import support
from undercurrent.adapters.vllm.worker_extension import _UNSUPPORTED_TENSOR_TYPES, ProbingWorkerExtension
from undercurrent.spec import TensorType


def test_support_module_does_not_import_vllm():
    import subprocess
    import sys

    code = "import sys, undercurrent.adapters.vllm.support; print('vllm' in sys.modules, 'torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"


def test_unsupported_types_match_worker():
    table = support.supported_tensor_types()
    assert set(table) == set(TensorType)
    assert {t for t, reason in table.items() if reason} == set(_UNSUPPORTED_TENSOR_TYPES)


def _tiny_models():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    configs = [
        transformers.GPT2Config(n_layer=2, n_embd=16, n_head=2),
        transformers.LlamaConfig(
            num_hidden_layers=2, hidden_size=16, num_attention_heads=2, intermediate_size=32, vocab_size=50
        ),
    ]
    with torch.device("meta"):
        return [transformers.AutoModelForCausalLM.from_config(c) for c in configs]


def test_model_support_matches_worker_hook_discovery():
    for model in _tiny_models():
        worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        layers = ProbingWorkerExtension._find_decoder_layers(worker)
        final_norm = ProbingWorkerExtension._find_final_norm(worker)
        table = support.model_tensor_support(model)
        assert table[TensorType.RESIDUAL_STREAM] is None
        assert (table[TensorType.ATTN_OUT] is None) == (getattr(layers[0], "self_attn", None) is not None)
        assert (table[TensorType.MLP_OUT] is None) == (getattr(layers[0], "mlp", None) is not None)
        assert (table[TensorType.FINAL_NORM] is None) == (final_norm is not None)


def test_model_without_layers():
    table = support.model_tensor_support(object())
    assert all(reason for reason in table.values())

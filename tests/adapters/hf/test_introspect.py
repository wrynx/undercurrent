"""undercurrent.adapters.hf.introspect: the helpers HFEngineAdapter hooks with."""

import pytest

from undercurrent.adapters.hf import HFAdapterLimitationError
from undercurrent.adapters.hf.adapter import _UNSUPPORTED_TENSOR_TYPES
from undercurrent.adapters.hf.introspect import (
    find_attention,
    find_decoder_layers,
    find_mlp,
    model_tensor_support,
    supported_tensor_types,
)
from undercurrent.spec import TensorType

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def meta_model(config):
    with torch.device("meta"):
        return transformers.AutoModelForCausalLM.from_config(config)


def test_supported_tensor_types_covers_every_type():
    table = supported_tensor_types()
    assert set(table) == set(TensorType)
    assert {t for t, reason in table.items() if reason} == {TensorType.KV, TensorType.FINAL_NORM}
    assert set(_UNSUPPORTED_TENSOR_TYPES) == {TensorType.KV, TensorType.FINAL_NORM}


def test_gpt2_layers():
    model = meta_model(transformers.GPT2Config(n_layer=3, n_embd=16, n_head=2))
    layers = find_decoder_layers(model)
    assert list(layers) == [0, 1, 2]
    assert layers[0] is model.transformer.h[0]
    assert find_attention(layers[0]) is model.transformer.h[0].attn
    assert find_mlp(layers[0]) is model.transformer.h[0].mlp
    assert model_tensor_support(model) == supported_tensor_types()


def test_llama_layers():
    config = transformers.LlamaConfig(
        num_hidden_layers=2, hidden_size=16, num_attention_heads=2, intermediate_size=32, vocab_size=50
    )
    model = meta_model(config)
    layers = find_decoder_layers(model)
    assert layers[1] is model.model.layers[1]
    assert find_attention(layers[1]) is model.model.layers[1].self_attn


def test_unknown_architecture_raises():
    with pytest.raises(HFAdapterLimitationError, match="find_decoder_layers"):
        find_decoder_layers(torch.nn.Linear(2, 2))


def test_missing_submodules_are_reported():
    class Block(torch.nn.Module):
        pass

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

    support = model_tensor_support(Model())
    assert "self_attn" in support[TensorType.ATTN_OUT]
    assert "mlp" in support[TensorType.MLP_OUT]
    assert support[TensorType.RESIDUAL_STREAM] is None

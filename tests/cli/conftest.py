"""Tiny model configs saved to disk, so `inspect-model` runs fully offline."""

import pytest


@pytest.fixture
def gpt2_dir(tmp_path):
    from transformers import GPT2Config

    path = tmp_path / "tiny-gpt2"
    GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=100).save_pretrained(path)
    return path


@pytest.fixture
def llama_dir(tmp_path):
    from transformers import LlamaConfig

    path = tmp_path / "tiny-llama"
    LlamaConfig(
        num_hidden_layers=4, hidden_size=32, num_attention_heads=2, intermediate_size=64, vocab_size=128
    ).save_pretrained(path)
    return path


@pytest.fixture
def opt_dir(tmp_path):
    """OPT keeps its decoder layers at model.decoder.layers, which neither adapter looks for."""
    from transformers import OPTConfig

    path = tmp_path / "tiny-opt"
    OPTConfig(num_hidden_layers=2, hidden_size=32, num_attention_heads=2, ffn_dim=64, vocab_size=100).save_pretrained(
        path
    )
    return path

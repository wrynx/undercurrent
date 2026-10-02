"""Shared fixtures for the HF adapter's tests (builders live in `_helpers.py`).

Builds a tiny, randomly-initialized GPT-2 model (2 layers, small hidden
size) and a matching minimal tokenizer -- no network access or pretrained
checkpoint download needed, mirroring content_safety_demo's own
"structural, random-initialized, not trained" convention for test fixtures.
"""

import pytest
import torch

from tests.adapters.hf._helpers import SlowThenContinueProbe, make_tiny_gpt2, make_tiny_tokenizer
from undercurrent.router import ProbeFactory, Router


@pytest.fixture
def tiny_model():
    torch.manual_seed(1234)
    return make_tiny_gpt2()


@pytest.fixture
def tiny_tokenizer():
    return make_tiny_tokenizer()


@pytest.fixture
def adapter(tiny_model, tiny_tokenizer):
    from undercurrent.adapters.hf import HFEngineAdapter

    a = HFEngineAdapter()
    a.load_model(tiny_model, tokenizer=tiny_tokenizer, device="cpu")
    return a


@pytest.fixture
def prompt():
    return "tok1 tok2 tok3"


@pytest.fixture
def probe_registry():
    from content_safety_demo import TrajectorySafetyProbe

    return {
        "trajectory_safety": ProbeFactory(TrajectorySafetyProbe, {"layer": 0, "seed": 7}),
        "slow_then_continue": ProbeFactory(SlowThenContinueProbe, {}),
    }


@pytest.fixture
def router(probe_registry):
    r = Router(probe_registry)
    yield r
    r.shutdown()

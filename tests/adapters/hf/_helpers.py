"""Test helpers shared by the HF adapter tests (not fixtures -- those live in
conftest.py). Import them absolutely: `from tests.adapters.hf._helpers import X`.

`make_tiny_gpt2` / `make_tiny_tokenizer` build a tiny, randomly-initialized
GPT-2 model (2 layers, small hidden size) and a matching minimal tokenizer --
no network access or pretrained checkpoint download needed, following the
content-safety demo's "structural, random-initialized, not trained"
convention for test fixtures.
"""

import time

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from undercurrent.core import Probe, ProbeResult, ProbeSignal
from undercurrent.router import Router

VOCAB_SIZE = 64


def make_tiny_gpt2():
    config = GPT2Config(
        n_layer=2,
        n_embd=16,
        n_head=2,
        n_positions=64,
        vocab_size=VOCAB_SIZE,
        n_inner=32,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def make_tiny_tokenizer():
    vocab = {f"tok{i}": i for i in range(VOCAB_SIZE)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="tok0")


class SlowThenContinueProbe(Probe):
    """Test double: on_activation sleeps well past any short timeout_ms,
    then returns action=CONTINUE -- used to prove a block_until_signal
    extraction point's on_timeout fallback fires because of the timeout
    itself, not because the probe would have asked to abort anyway (it
    never would, if given enough time)."""

    probe_kind = "trajectory"

    def __init__(self, delay: float = 2.0) -> None:
        super().__init__()
        self._delay = delay
        self.received = []

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        self.received.append(record)
        time.sleep(self._delay)
        return ProbeSignal()  # CONTINUE, but arrives far too late to matter

    def on_end(self, request_ctx):
        return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=list(self.received))


def spy_end_request(router: Router):
    """Wrap `router.end_request` to stash its return value -- needed
    because `HFEngineAdapter.generate()` calls `end_request` internally
    (per the `EngineAdapter` contract, `generate()` only returns `str`), so
    a test that wants the `ProbeResult`s has nowhere else to get them.
    Returns a dict that gets populated the first time end_request runs."""
    captured = {}
    original = router.end_request

    def _spy(request_id):
        result = original(request_id)
        captured["results"] = result
        return result

    router.end_request = _spy
    return captured

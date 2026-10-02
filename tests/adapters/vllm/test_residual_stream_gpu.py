"""GPU check: vLLM's `residual_stream` equals the HF backend's, token by token.

On vLLM's fused-residual decoder layers (Llama and most modern families) the
residual stream after a layer is `hidden_states + residual`; see
`_residual_stream_from_layer_output()` in
`undercurrent/adapters/vllm/worker_extension.py`. The CPU tests in
`test_residual_stream_capture.py` check that against fake layers. This test
checks it against vLLM itself: a tiny random Llama (fused residual) and a
tiny random GPT-2 (single tensor, which must stay as it was) are saved under
the pytest temp dir (so `$TMPDIR`), loaded in float32 by the HF and the vLLM
backend, run on the same prompt with greedy decoding, and the
`residual_stream` captures at the first and last layer must match per token.

Marked `gpu`: skipped without CUDA, and run by `scripts/gpu_check.sh` (the
pre-release GPU gate in RELEASING.md). Nothing is downloaded: the models and
the word-level tokenizer are built locally.
"""

from __future__ import annotations

import gc

import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("vllm", reason="needs a real vLLM install")

import torch  # noqa: E402

from tests.adapters.vllm._helpers import make_extraction_point  # noqa: E402
from undercurrent.core import Probe, ProbeResult, RequestContext  # noqa: E402
from undercurrent.router import ProbeFactory, Router  # noqa: E402
from undercurrent.spec import ProbeKind  # noqa: E402

NUM_LAYERS = 3
LAYERS = (0, NUM_LAYERS - 1)
WORDS = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "a",
    "lazy",
    "dog",
    "and",
    "then",
    "runs",
    "far",
    "away",
    "into",
    "woods",
]
PROMPT = "the quick brown fox jumps over the lazy dog"
VOCAB_SIZE = 64
# float32 on both sides; the slack covers kernel differences (fused vs. separate
# norm/add, attention backends). The bug this guards against was off by the
# whole residual, i.e. a relative error of order 1.
MAX_RELATIVE_ERROR = 1e-3


class SinkProbe(Probe):
    probe_kind = "trajectory"

    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def on_activation(self, record):
        self._sink.append(record)
        return None

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=None)


def _save_tokenizer(path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({word: i + len(vocab) for i, word in enumerate(WORDS)})
    assert len(vocab) <= VOCAB_SIZE
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<unk>"
    )
    tokenizer.save_pretrained(path)


def _save_tiny_model(path, architecture: str) -> None:
    import transformers

    torch.manual_seed(0)
    common = {"vocab_size": VOCAB_SIZE, "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float32"}
    if architecture == "llama":
        config = transformers.LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=False,
            **common,
        )
        model = transformers.LlamaForCausalLM(config)
    else:
        config = transformers.GPT2Config(n_embd=256, n_layer=NUM_LAYERS, n_head=4, n_positions=128, **common)
        model = transformers.GPT2LMHeadModel(config)
    model.save_pretrained(path)
    _save_tokenizer(path)


def _extraction_points(num_prompt_tokens: int):
    # One extraction point per prompt token (absolute position), each on both layers.
    return [
        make_extraction_point(
            name=f"pos-{pos}", layer=LAYERS, position=str(pos), probe_type="sink", probe_kind=ProbeKind.TRAJECTORY
        )
        for pos in range(num_prompt_tokens)
    ]


def _num_prompt_tokens(path) -> int:
    from transformers import AutoTokenizer

    return len(AutoTokenizer.from_pretrained(path)(PROMPT)["input_ids"])


def _capture(adapter, request_id: str, generation_kwargs: dict, num_prompt_tokens: int) -> dict:
    """`{(layer, token_pos): tensor}` for every prompt token at LAYERS."""
    sink: list = []
    router = Router(probe_registry={"sink": ProbeFactory(SinkProbe, {"sink": sink})})
    try:
        adapter.register_extraction(request_id, _extraction_points(num_prompt_tokens))
        adapter.generate(request_id, PROMPT, generation_kwargs, router)
    finally:
        router.shutdown(wait=True)
    captured = {(r.layer, r.token_pos): r.tensor.detach().to("cpu", torch.float32).flatten() for r in sink}
    assert set(captured) == {(layer, pos) for layer in LAYERS for pos in range(num_prompt_tokens)}
    return captured


def _hf_capture(path, num_prompt_tokens: int) -> dict:
    from undercurrent.adapters.hf import HFEngineAdapter

    adapter = HFEngineAdapter()
    adapter.load_model(str(path), device="cuda", torch_dtype=torch.float32)
    try:
        return _capture(adapter, "hf", {"max_new_tokens": 2, "do_sample": False, "pad_token_id": 0}, num_prompt_tokens)
    finally:
        adapter.close()
        del adapter
        gc.collect()
        torch.cuda.empty_cache()


def _vllm_capture(path, num_prompt_tokens: int) -> dict:
    from undercurrent.adapters.vllm import VLLMEngineAdapter

    adapter = VLLMEngineAdapter()
    adapter.load_model(str(path), dtype="float32", gpu_memory_utilization=0.3, max_model_len=64, enforce_eager=True)
    try:
        return _capture(adapter, "vllm", {"max_tokens": 2, "temperature": 0.0}, num_prompt_tokens)
    finally:
        adapter.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "architecture",
    [
        pytest.param("llama", id="llama-fused-residual"),
        pytest.param("gpt2", id="gpt2-single-tensor"),
    ],
)
def test_vllm_residual_stream_matches_hf(tmp_path_factory, architecture):
    path = tmp_path_factory.mktemp(f"tiny-{architecture}")
    _save_tiny_model(path, architecture)
    num_prompt_tokens = _num_prompt_tokens(path)

    hf = _hf_capture(path, num_prompt_tokens)
    vllm = _vllm_capture(path, num_prompt_tokens)

    for key in sorted(hf):
        expected, actual = hf[key], vllm[key]
        relative_error = float((actual - expected).norm() / expected.norm())
        assert relative_error <= MAX_RELATIVE_ERROR, (
            f"{architecture} layer={key[0]} token={key[1]}: vLLM residual_stream differs from HF's "
            f"(relative error {relative_error:.3g})"
        )

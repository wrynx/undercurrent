"""Tests for `undercurrent.model.ProbedModel` with the vLLM backend.

No vLLM or GPU needed: `undercurrent.model._new_vllm_adapter` is replaced by
a fake `EngineAdapter` that routes synthetic `ActivationRecord`s through the
router, sleeps briefly, and records how many `generate()` calls overlap.
The one real-vLLM smoke test at the bottom is marked `gpu` and `network`.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

import undercurrent.model as model_mod
from undercurrent.adapters._optional import MissingDependencyError
from undercurrent.adapters.base import EngineAdapter
from undercurrent.core import Probe, ProbeAction, ProbeResult, ProbeSignal, RequestContext
from undercurrent.model import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_VLLM_MAX_CONCURRENCY,
    GenerationOutput,
    ProbedModel,
    ProbedModelConfigError,
    VLLMBackend,
)
from undercurrent.spec import ActivationRecord, parse_dict

NUM_LAYERS = 4


class CountProbe(Probe):
    probe_kind = "single_shot"

    def on_start(self, ctx):
        self.count = 0

    def on_activation(self, record):
        self.count += 1

    def on_end(self, ctx):
        return ProbeResult(ctx.request_id, self.extraction_point_name, verdict=self.count)


class AbortProbe(Probe):
    """Aborts on its first activation when the prompt contains "bad"."""

    probe_kind = "trajectory"

    def on_start(self, ctx):
        self.bad = "bad" in ctx.prompt_metadata["prompt"]
        self.history = []

    def on_activation(self, record):
        if not self.bad:
            return None
        signal = ProbeSignal(action=ProbeAction.ABORT, confidence=0.9)
        self.history.append(signal)
        return signal

    def on_end(self, ctx):
        return ProbeResult(ctx.request_id, self.extraction_point_name, verdict=self.bad, signal_history=self.history)


PROBES = {"count": CountProbe, "abort": AbortProbe}

SPEC = parse_dict(
    {
        "version": "1",
        "extraction_points": [
            {
                "name": "count",
                "probe_type": "count",
                "probe_kind": "single_shot",
                "tensor_type": "residual_stream",
                "layers": [1],
                "position": "prompt[-1]",
                "execution_mode": "inline",
            },
            {
                "name": "guard",
                "probe_type": "abort",
                "probe_kind": "trajectory",
                "tensor_type": "mlp_out",
                "layers": [2],
                "position": "generated[*]",
                "execution_mode": "inline",
            },
        ],
    }
)


class FakeVLLMAdapter(EngineAdapter):
    """Stands in for `VLLMEngineAdapter`: same calling pattern (register,
    then a blocking `generate()` that routes records and ends the request),
    safe to call from many threads at once."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.loaded = None
        self.pending = {}
        self.kwargs_seen = []
        self.unregistered = []
        self.shutdown_calls = 0
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def load_model(self, model_name_or_path, **kwargs):
        self.loaded = (model_name_or_path, kwargs)

    def register_extraction(self, request_id, extraction_points):
        self.pending[request_id] = list(extraction_points)

    def generate(self, request_id, prompt, generation_kwargs, router):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.kwargs_seen.append(dict(generation_kwargs))
        points = self.pending.pop(request_id)
        router.register_request(request_id, points, RequestContext(request_id, {"prompt": prompt}, None))
        try:
            if prompt.startswith("fail"):
                raise RuntimeError(f"engine error for {prompt}")
            for ep in points:
                router.route(
                    ActivationRecord(
                        request_id=request_id,
                        extraction_point_name=ep.name,
                        layer=ep.layers[0],
                        token_pos=0,
                        tensor_type=ep.tensor_type.value,
                        tensor=[0.0],
                        is_generated=False,
                    )
                )
            time.sleep(self.delay)
            return f"out:{prompt}"
        finally:
            router.end_request(request_id)
            with self._lock:
                self.active -= 1

    def unregister_extraction(self, request_id):
        self.unregistered.append(request_id)

    def shutdown(self):
        self.shutdown_calls += 1


@pytest.fixture
def fake(monkeypatch):
    adapter = FakeVLLMAdapter()
    monkeypatch.setattr(model_mod, "_new_vllm_adapter", lambda: adapter)
    monkeypatch.setattr(model_mod, "_config_num_layers", lambda *a, **k: NUM_LAYERS)
    return adapter


@pytest.fixture
def make_model(fake):
    built = []

    def _make(**kwargs):
        kwargs.setdefault("spec", SPEC)
        kwargs.setdefault("probes", PROBES)
        m = ProbedModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", backend="vllm", **kwargs)
        built.append(m)
        return m

    yield _make
    for m in built:
        m.close()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_engine_args_reach_load_model(make_model, fake):
    m = make_model(gpu_memory_utilization=0.85, max_model_len=4096)
    assert m.adapter is fake
    assert fake.loaded == ("meta-llama/Llama-3.1-8B-Instruct", {"gpu_memory_utilization": 0.85, "max_model_len": 4096})
    assert m.max_concurrency == DEFAULT_VLLM_MAX_CONCURRENCY
    assert repr(m).startswith("ProbedModel(backend='vllm'")


def test_single_prompt_and_abort_from_signal_history(make_model):
    m = make_model()
    ok = m.generate("hello")
    bad = m.generate("bad prompt")

    assert isinstance(ok, GenerationOutput)
    assert ok.text == "out:hello"
    assert ok.probe_results["count"].verdict == 1
    assert not ok.aborted
    # The vLLM adapter doesn't report its stop signal: the abort comes from
    # the inline point's signal_history.
    assert bad.aborted and bad.abort_point == "guard"
    assert bad.abort_signal.confidence == 0.9


def test_device_must_be_cuda_or_none(fake, monkeypatch):
    with pytest.raises(ProbedModelConfigError, match="device='cpu'"):
        ProbedModel.from_pretrained("m", backend="vllm", spec=SPEC, probes=PROBES, device="cpu")
    with ProbedModel.from_pretrained("m", backend="vllm", spec=SPEC, probes=PROBES, device="cuda") as m:
        assert m.generate("x").text == "out:x"
    assert "device" not in fake.loaded[1]


def test_model_object_is_rejected(fake):
    with pytest.raises(TypeError, match="hub id or local path"):
        ProbedModel.from_pretrained(object(), backend="vllm", spec=SPEC, probes=PROBES)


def test_kv_tensor_type_rejected_before_load(fake):
    spec = {
        "version": "1",
        "extraction_points": [
            {
                "name": "kv",
                "probe_type": "count",
                "probe_kind": "single_shot",
                "tensor_type": "kv",
                "layers": [0],
                "position": "prompt[-1]",
                "execution_mode": "inline",
            }
        ],
    }
    with pytest.raises(ProbedModelConfigError, match="'kv' is not supported by the 'vllm' backend"):
        ProbedModel.from_pretrained("m", backend="vllm", spec=spec, probes=PROBES)
    assert fake.loaded is None


def test_layer_out_of_range_fails_before_load(fake, monkeypatch):
    monkeypatch.setattr(model_mod, "_config_num_layers", lambda *a, **k: 2)
    with pytest.raises(ProbedModelConfigError, match="layer 2 is out of range"):
        ProbedModel.from_pretrained("m", backend="vllm", spec=SPEC, probes=PROBES)
    assert fake.loaded is None


def test_num_layers_read_from_local_config_without_weights(tmp_path):
    from transformers import GPT2Config

    GPT2Config(n_layer=3).save_pretrained(tmp_path)
    assert model_mod._config_num_layers(str(tmp_path), revision=None, trust_remote_code=False) == 3
    assert VLLMBackend().config_num_layers(tmp_path, {}) == 3
    # Unreadable config: no layer check rather than a failure.
    assert model_mod._config_num_layers(str(tmp_path / "missing"), revision=None, trust_remote_code=False) is None


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_batch_is_concurrent_and_order_preserved(make_model, fake):
    m = make_model(max_concurrency=4)
    prompts = [f"p{i}" for i in range(12)]
    outs = m.generate(prompts)

    assert [o.text for o in outs] == [f"out:{p}" for p in prompts]
    assert [o.prompt for o in outs] == prompts
    assert len({o.request_id for o in outs}) == len(prompts)
    assert all(o.probe_results["count"].verdict == 1 for o in outs)
    assert fake.peak == 4
    assert sorted(fake.unregistered) == sorted(o.request_id for o in outs)


def test_concurrency_limit_holds_across_caller_threads(make_model, fake):
    m = make_model(max_concurrency=3)
    threads = [threading.Thread(target=m.generate, args=([f"t{i}-{j}" for j in range(4)],)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.peak == 3


def test_batch_of_one_and_empty_batch(make_model):
    m = make_model()
    assert [o.text for o in m.generate(["only"])] == ["out:only"]
    assert m.generate([]) == []


def test_max_concurrency_one_runs_sequentially(make_model, fake):
    m = make_model(max_concurrency=1)
    m.generate(["a", "b", "c"])
    assert fake.peak == 1


def test_default_max_concurrency_on_backend():
    assert VLLMBackend().max_concurrency == DEFAULT_VLLM_MAX_CONCURRENCY
    assert VLLMBackend(max_concurrency=8).max_concurrency == 8


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_max_concurrency(fake, bad):
    with pytest.raises(ValueError, match="max_concurrency"):
        ProbedModel.from_pretrained("m", backend="vllm", spec=SPEC, probes=PROBES, max_concurrency=bad)


def test_hf_backend_rejects_concurrency_above_one():
    with pytest.raises(ProbedModelConfigError, match="'hf' backend"):
        ProbedModel.from_pretrained("openai-community/gpt2", max_concurrency=2)


def test_one_failure_raises_for_the_whole_call(make_model, fake):
    m = make_model(max_concurrency=2)
    with pytest.raises(RuntimeError, match="engine error for fail-1"):
        m.generate(["a", "fail-1", "b", "c", "d", "e", "f", "g"])
    # Prompts that hadn't started yet were cancelled.
    assert len(fake.kwargs_seen) < 8
    # Every started request was cleaned up.
    assert len(fake.unregistered) == len(fake.kwargs_seen)


def test_return_exceptions_puts_error_in_its_slot(make_model):
    m = make_model(max_concurrency=3)
    outs = m.generate(["a", "fail-1", "b", "fail-2"], return_exceptions=True)

    assert outs[0].text == "out:a" and outs[2].text == "out:b"
    assert isinstance(outs[1], RuntimeError) and "fail-1" in str(outs[1])
    assert isinstance(outs[3], RuntimeError) and "fail-2" in str(outs[3])


def test_return_exceptions_sequential_path(make_model):
    m = make_model(max_concurrency=1)
    outs = m.generate(["fail-x", "a"], return_exceptions=True)
    assert isinstance(outs[0], RuntimeError)
    assert outs[1].text == "out:a"
    with pytest.raises(RuntimeError):
        m.generate(["fail-x", "a"])


def test_on_result_called_from_worker_threads(fake):
    seen = []
    with ProbedModel.from_pretrained(
        "m", backend="vllm", spec=SPEC, probes=PROBES, on_result=seen.append, max_concurrency=4
    ) as m:
        outs = m.generate([f"p{i}" for i in range(6)])
    assert sorted(o.request_id for o in seen) == sorted(o.request_id for o in outs)


# ---------------------------------------------------------------------------
# Kwarg translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({}, {"max_tokens": DEFAULT_MAX_NEW_TOKENS}),
        ({"max_new_tokens": 64}, {"max_tokens": 64}),
        ({"max_tokens": 10}, {"max_tokens": 10}),
        ({"min_new_tokens": 4, "max_new_tokens": 8}, {"min_tokens": 4, "max_tokens": 8}),
        (
            {"temperature": 0.0, "top_p": 0.9, "seed": 7},
            {"temperature": 0.0, "top_p": 0.9, "seed": 7, "max_tokens": DEFAULT_MAX_NEW_TOKENS},
        ),
        ({"stop": "\n"}, {"stop": ["\n"], "max_tokens": DEFAULT_MAX_NEW_TOKENS}),
        ({"stop": ("a", "b")}, {"stop": ["a", "b"], "max_tokens": DEFAULT_MAX_NEW_TOKENS}),
        ({"repetition_penalty": 1.1}, {"repetition_penalty": 1.1, "max_tokens": DEFAULT_MAX_NEW_TOKENS}),
    ],
)
def test_vllm_kwarg_translation(given, expected):
    assert VLLMBackend().translate_kwargs(given) == expected


def test_generate_translates_kwargs(make_model, fake):
    m = make_model()
    m.generate("x", max_new_tokens=64, temperature=0.7, top_p=0.95, seed=3, stop="END")
    assert fake.kwargs_seen == [{"max_tokens": 64, "temperature": 0.7, "top_p": 0.95, "seed": 3, "stop": ["END"]}]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_shuts_the_adapter_down(make_model, fake):
    m = make_model()
    m.close()
    m.close()
    assert fake.shutdown_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        m.generate("x")


def test_close_after_failed_validation_shuts_down(fake, monkeypatch):
    monkeypatch.setattr(model_mod, "_config_num_layers", lambda *a, **k: None)
    monkeypatch.setattr(VLLMBackend, "num_layers", lambda self, adapter: 2)
    with pytest.raises(ProbedModelConfigError):
        ProbedModel.from_pretrained("m", backend="vllm", spec=SPEC, probes=PROBES)
    assert fake.shutdown_calls == 1


def test_caller_owned_vllm_adapter_gets_vllm_backend_and_is_not_shut_down(monkeypatch):
    from undercurrent.adapters.vllm import VLLMEngineAdapter

    class OwnedFake(FakeVLLMAdapter, VLLMEngineAdapter):
        def __init__(self):
            FakeVLLMAdapter.__init__(self)

    adapter = OwnedFake()
    with ProbedModel.from_pretrained(None, backend=adapter, spec=SPEC, probes=PROBES) as m:
        assert m.max_concurrency == DEFAULT_VLLM_MAX_CONCURRENCY
        m.generate("x", max_new_tokens=5)
    assert adapter.kwargs_seen == [{"max_tokens": 5}]
    assert adapter.shutdown_calls == 0


# ---------------------------------------------------------------------------
# Missing / unsupported vLLM
# ---------------------------------------------------------------------------


def test_missing_vllm_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setattr(model_mod, "_config_num_layers", lambda *a, **k: None)
    with pytest.raises(MissingDependencyError) as info:
        ProbedModel.from_pretrained("m", backend="vllm", spec=SPEC, probes=PROBES)
    message = str(info.value)
    assert "vLLM must be installed in this environment" in message
    assert "CUDA" in message and "torch" in message
    assert "existing vLLM environment or image" in message
    assert 'pip install "undercurrent[vllm]"' in message
    assert "docs/compatibility.md" in message
    assert isinstance(info.value, ImportError)


def test_unsupported_vllm_version_uses_the_adapter_check(monkeypatch):
    from undercurrent.adapters.vllm import VLLMAdapterLimitationError, version_check

    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setattr(version_check, "_installed_vllm_version", lambda: "0.1.0")
    monkeypatch.delenv(version_check.ALLOW_UNSUPPORTED_VLLM_ENV, raising=False)
    with pytest.raises(VLLMAdapterLimitationError, match=r"vllm==0\.1\.0 is installed"):
        model_mod._new_vllm_adapter()


def test_import_model_does_not_import_vllm():
    import subprocess

    code = "import sys, undercurrent.model; assert 'vllm' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------------
# Real vLLM (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.slow
def test_real_vllm_smoke():
    pytest.importorskip("vllm", reason="needs a real vLLM install")
    with ProbedModel.from_pretrained(
        "openai-community/gpt2",
        backend="vllm",
        spec=SPEC,
        probes=PROBES,
        gpu_memory_utilization=0.3,
        max_model_len=128,
        enforce_eager=True,
    ) as m:
        outs = m.generate(["Hello, my name is", "The capital of France is", "bad"], max_new_tokens=8, temperature=0)
    assert [o.prompt for o in outs] == ["Hello, my name is", "The capital of France is", "bad"]
    assert all(isinstance(o.text, str) for o in outs)
    assert outs[0].probe_results["count"].verdict >= 1
    assert not outs[0].aborted
    assert outs[2].aborted

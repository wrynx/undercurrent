"""Tests for `undercurrent.model.ProbedModel` with the HF backend.

CPU-only and offline: every test runs on the tiny random GPT-2 from
`tests/adapters/hf/_helpers.py` (2 layers, vocab 64) with an offline
WordLevel tokenizer passed as `tokenizer=`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from tests.adapters.hf._helpers import make_tiny_gpt2, make_tiny_tokenizer
from undercurrent.adapters.base import EngineAdapter
from undercurrent.core import Probe, ProbeAction, ProbeFactory, ProbeResult, ProbeSignal, RequestContext
from undercurrent.model import (
    DEFAULT_MAX_NEW_TOKENS,
    GenerationOutput,
    HFBackend,
    ProbedModel,
    ProbedModelConfigError,
)
from undercurrent.router import MetricsSink, OverflowPolicy, Router, RouterError
from undercurrent.spec import ActivationRecord, parse_dict

PROMPT = "tok1 tok2 tok3"
# min_new_tokens == max_new_tokens keeps the random model's EOS from ending
# generation early, so any early stop is an abort's doing.
FIXED = {"max_new_tokens": 8, "min_new_tokens": 8, "temperature": 0.0}


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


class CountProbe(Probe):
    """Single-shot probe; verdict is the number of activations it saw."""

    probe_kind = "single_shot"

    def on_start(self, ctx):
        self.count = 0

    def on_activation(self, record):
        self.count += 1

    def on_end(self, ctx):
        return ProbeResult(ctx.request_id, self.extraction_point_name, verdict=self.count)


class AbortProbe(Probe):
    """Trajectory probe that aborts on its first activation."""

    probe_kind = "trajectory"

    def on_start(self, ctx):
        self.history = []

    def on_activation(self, record):
        signal = ProbeSignal(action=ProbeAction.ABORT, confidence=0.93)
        self.history.append(signal)
        return signal

    def on_end(self, ctx):
        return ProbeResult(ctx.request_id, self.extraction_point_name, verdict="toxic", signal_history=self.history)


class TokenTrajectoryProbe(Probe):
    """Trajectory probe recording the token positions it saw (for async points)."""

    probe_kind = "trajectory"

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self._delay = delay

    def on_start(self, ctx):
        self.positions = []

    def on_activation(self, record):
        if self._delay:
            time.sleep(self._delay)
        self.positions.append(record.token_pos)
        return ProbeSignal(action=ProbeAction.FLAG)

    def on_end(self, ctx):
        return ProbeResult(ctx.request_id, self.extraction_point_name, verdict=list(self.positions))


PROBES = {"count": CountProbe, "abort": AbortProbe, "trajectory": TokenTrajectoryProbe}


def point(name="last_prompt", probe_type="count", probe_kind="single_shot", **overrides):
    return {
        "name": name,
        "layers": 1,
        "tensor_type": "residual_stream",
        "position": "prompt[-1]",
        "probe_type": probe_type,
        "probe_kind": probe_kind,
        **overrides,
    }


def spec_of(*points):
    return {"version": "1", "extraction_points": list(points) or [point()]}


ABORT_POINT = point(name="toxicity", probe_type="abort", probe_kind="trajectory", position="generated[2:]")
ASYNC_POINT = point(
    name="observer",
    probe_type="trajectory",
    probe_kind="trajectory",
    tensor_type="mlp_out",
    layers=[0, 1],
    position="generated[*]",
    execution_mode="async",
)


@pytest.fixture
def make_model():
    built = []

    def _make(spec=None, **kwargs):
        torch.manual_seed(1234)
        kwargs.setdefault("probes", PROBES)
        m = ProbedModel.from_pretrained(
            make_tiny_gpt2(), tokenizer=make_tiny_tokenizer(), spec=spec if spec is not None else spec_of(), **kwargs
        )
        built.append(m)
        return m

    yield _make
    for m in built:
        m.close()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_single_prompt(make_model):
    m = make_model()
    out = m.generate(PROMPT, **FIXED)

    assert isinstance(out, GenerationOutput)
    assert out.prompt == PROMPT
    assert len(out.text.split()) == 8
    assert out.request_id
    assert not out.aborted
    assert out.abort_reason is None and out.abort_signal is None and out.abort_point is None
    assert set(out.probe_results) == {"last_prompt"}
    assert out.probe_results["last_prompt"].verdict == 1
    assert out.probe_results["last_prompt"].request_id == out.request_id


def test_batch_returns_outputs_in_order(make_model):
    m = make_model()
    prompts = ["tok1", "tok2 tok3", "tok4 tok5 tok6"]
    outs = m.generate(prompts, max_new_tokens=4, min_new_tokens=4)

    assert isinstance(outs, list)
    assert [o.prompt for o in outs] == prompts
    assert len({o.request_id for o in outs}) == 3
    assert all(o.probe_results["last_prompt"].verdict == 1 for o in outs)


def test_hf_backend_runs_one_generation_at_a_time(make_model):
    m = make_model()
    assert m.max_concurrency == 1

    # Concurrent callers are serialised instead of hitting the adapter's
    # "one in-flight generate()" limitation.
    outs = []
    threads = [threading.Thread(target=lambda: outs.append(m.generate(PROMPT, max_new_tokens=3))) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(outs) == 3


def test_inline_probe_aborts_generation(make_model):
    m = make_model(spec_of(point(), ABORT_POINT))
    out = m.generate(PROMPT, **FIXED)

    assert out.aborted
    assert out.abort_point == "toxicity"
    assert out.abort_signal is not None and out.abort_signal.action == ProbeAction.ABORT
    assert out.abort_reason == "extraction point 'toxicity' aborted generation (confidence 0.93)"
    # Aborted at generated token 2: fewer than the 8 forced tokens.
    assert len(out.text.split()) < 8
    assert out.probe_results["toxicity"].verdict == "toxic"
    assert "aborted=True" in repr(out)


def test_async_probe_results_present(make_model):
    m = make_model(spec_of(point(), ASYNC_POINT))
    out = m.generate(PROMPT, **FIXED)

    positions = out.probe_results["observer"].verdict
    # Two layers per generated token. The prefill step yields generated
    # token 0 without a decode-step forward, so 7 of the 8 tokens are seen.
    assert len(positions) == 2 * 7
    assert not out.aborted  # FLAG signals from an async point never abort


def test_generation_output_repr_truncates_text():
    out = GenerationOutput(
        text="x" * 500,
        prompt="p",
        request_id="r",
        probe_results={"a": ProbeResult("r", "a", verdict=0.5)},
    )
    text = repr(out)
    assert len(text) < 200
    assert "a: verdict=0.5" in text
    with pytest.raises(AttributeError):
        out.text = "changed"  # frozen


# ---------------------------------------------------------------------------
# Spec input forms
# ---------------------------------------------------------------------------

SPEC_YAML = """\
version: "1"
extraction_points:
  - name: last_prompt
    layers: 1
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: count
    probe_kind: single_shot
"""


def _spec_forms(tmp_path):
    path = tmp_path / "probes.yaml"
    path.write_text(SPEC_YAML, encoding="utf-8")
    parsed = parse_dict(spec_of())
    return {
        "path_str": str(path),
        "pathlike": path,
        "yaml_str": SPEC_YAML,
        "dict": spec_of(),
        "probespec": parsed,
        "list": list(parsed.extraction_points),
    }


@pytest.mark.parametrize("form", ["path_str", "pathlike", "yaml_str", "dict", "probespec", "list"])
def test_every_spec_input_form(make_model, tmp_path, form):
    m = make_model(_spec_forms(tmp_path)[form])
    assert m.spec.names == ("last_prompt",)
    out = m.generate(PROMPT, max_new_tokens=2)
    assert out.probe_results["last_prompt"].verdict == 1


def test_no_spec_means_no_probes():
    torch.manual_seed(0)
    with ProbedModel.from_pretrained(make_tiny_gpt2(), tokenizer=make_tiny_tokenizer()) as m:
        out = m.generate(PROMPT, max_new_tokens=2)
    assert out.probe_results == {}
    assert not out.aborted


def test_missing_spec_file_raises(make_model):
    with pytest.raises(FileNotFoundError):
        make_model("does/not/exist.yaml")


# ---------------------------------------------------------------------------
# Validation (fail fast, at construction)
# ---------------------------------------------------------------------------


def test_unknown_probe_type(make_model):
    with pytest.raises(ProbedModelConfigError) as exc:
        make_model(spec_of(point(name="mine", probe_type="cout")))
    message = str(exc.value)
    assert "'mine'" in message
    assert "'count'" in message  # did-you-mean
    assert "probes={'cout': YourProbeClass}" in message


def test_probe_kind_mismatch(make_model):
    with pytest.raises(ProbedModelConfigError, match=r"'mine'.*probe_kind"):
        make_model(spec_of(point(name="mine", probe_type="abort", probe_kind="single_shot")))


def test_layer_out_of_range(make_model):
    with pytest.raises(ProbedModelConfigError, match=r"'deep'.*layer 5.*2 layers.*0\.\.1"):
        make_model(spec_of(point(name="deep", layers=[0, 5])))


def test_unsupported_tensor_type(make_model):
    with pytest.raises(ProbedModelConfigError, match=r"'cache'.*tensor_type='kv'.*'hf' backend.*residual_stream"):
        make_model(spec_of(point(name="cache", tensor_type="kv")))


def test_validation_failure_shuts_down_owned_router(make_model, monkeypatch):
    created = []
    original_init = Router.__init__

    def spy_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(Router, "__init__", spy_init)
    with pytest.raises(ProbedModelConfigError):
        make_model(spec_of(point(probe_type="nope")))
    assert created and created[0]._closed


def test_unknown_backend_name():
    with pytest.raises(ValueError, match="unknown backend 'tgi'"):
        ProbedModel.from_pretrained(make_tiny_gpt2(), tokenizer=make_tiny_tokenizer(), backend="tgi")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_context_manager_closes_router_and_adapter():
    torch.manual_seed(0)
    model = make_tiny_gpt2()
    with ProbedModel.from_pretrained(model, tokenizer=make_tiny_tokenizer(), spec=spec_of(), probes=PROBES) as m:
        router = m.router
        assert m.generate(PROMPT, max_new_tokens=2).probe_results
        assert "forward" in vars(model)  # the adapter's step-counting wrapper

    assert "forward" not in vars(model)  # hooks and wrapper removed
    with pytest.raises(RuntimeError, match="closed"):
        m.generate(PROMPT)
    with pytest.raises(RouterError, match="shut down"):
        router.register_request("r", [], RequestContext("r", {}, None))
    m.close()  # idempotent


def test_caller_owned_router_is_not_shut_down():
    router = Router(PROBES)
    try:
        torch.manual_seed(0)
        m = ProbedModel.from_pretrained(
            make_tiny_gpt2(), tokenizer=make_tiny_tokenizer(), spec=spec_of(), router=router
        )
        assert m.router is router
        assert m.generate(PROMPT, max_new_tokens=2).probe_results["last_prompt"].verdict == 1
        m.close()

        assert not router._closed
        assert router._request_end_listeners == []  # our listener was removed
        with router.request(parse_dict(spec_of())) as req:
            pass
        assert req.results["last_prompt"].verdict == 0
    finally:
        router.shutdown()


@pytest.mark.parametrize("extra", [{"probes": PROBES}, {"router_kwargs": {"drain_timeout": 1.0}}])
def test_router_conflicts_with_probes_and_router_kwargs(extra):
    router = Router(PROBES)
    try:
        with pytest.raises(ValueError, match="router= can't be combined"):
            ProbedModel.from_pretrained(make_tiny_gpt2(), tokenizer=make_tiny_tokenizer(), router=router, **extra)
    finally:
        router.shutdown()


# ---------------------------------------------------------------------------
# on_result
# ---------------------------------------------------------------------------


def test_on_result_called_once_per_prompt(make_model):
    seen = []
    m = make_model(on_result=seen.append)
    outs = m.generate(["tok1", "tok2"], max_new_tokens=2)
    out = m.generate(PROMPT, max_new_tokens=2)

    assert seen == [*outs, out]
    assert all(s is o for s, o in zip(seen, [*outs, out]))


def test_raising_on_result_is_logged_and_output_kept(make_model, caplog):
    def broken(output):
        raise RuntimeError("callback bug")

    m = make_model(on_result=broken)
    with caplog.at_level(logging.ERROR, logger="undercurrent"):
        outs = m.generate(["tok1", "tok2"], max_new_tokens=2)

    assert len(outs) == 2 and all(o.probe_results for o in outs)
    records = [r for r in caplog.records if r.name == "undercurrent" and "on_result" in r.getMessage()]
    assert len(records) == 2
    assert records[0].exc_info is not None and "callback bug" in str(records[0].exc_info[1])


# ---------------------------------------------------------------------------
# Advanced pass-throughs
# ---------------------------------------------------------------------------


class RecordingMetricsSink(MetricsSink):
    def __init__(self):
        self.drops = 0
        self.activations = 0
        self._lock = threading.Lock()

    def record_queue_depth(self, request_id, extraction_point_name, depth):
        pass

    def record_drop(self, request_id, extraction_point_name):
        with self._lock:
            self.drops += 1

    def record_activation(self, request_id, extraction_point_name, latency_seconds):
        with self._lock:
            self.activations += 1

    def record_probe_error(self, request_id, extraction_point_name):
        pass


class RecordingLogSink:
    def __init__(self):
        self.signals = []
        self.results = []

    def write_signal(self, request_id, extraction_point_name, signal):
        self.signals.append((request_id, extraction_point_name, signal))

    def write_result(self, request_id, extraction_point_name, result):
        self.results.append((request_id, extraction_point_name, result))


def test_metrics_sink_receives_records(make_model):
    sink = RecordingMetricsSink()
    m = make_model(spec_of(ASYNC_POINT), metrics_sink=sink)
    m.generate(PROMPT, **FIXED)
    assert sink.activations == 2 * 7


def test_metrics_sink_given_twice_is_rejected(make_model):
    with pytest.raises(ValueError, match="metrics_sink"):
        make_model(metrics_sink=RecordingMetricsSink(), router_kwargs={"metrics_sink": RecordingMetricsSink()})


def test_router_kwargs_reach_the_router(make_model):
    # A slow async probe behind a 1-slot queue that drops new items: most
    # activations must be dropped, which only happens if both
    # default_queue_depth and default_overflow_policy reached the Router.
    sink = RecordingMetricsSink()
    m = make_model(
        spec_of(ASYNC_POINT),
        probes={"trajectory": ProbeFactory(TokenTrajectoryProbe, {"delay": 0.05})},
        metrics_sink=sink,
        router_kwargs={
            "default_queue_depth": 1,
            "default_overflow_policy": OverflowPolicy.DROP_NEWEST,
            "worker_pool_size": 2,
        },
    )
    out = m.generate(PROMPT, **FIXED)

    assert m.router.worker_pool_size == 2
    assert sink.drops > 0
    assert len(out.probe_results["observer"].verdict) + sink.drops == 2 * 7


def test_log_sink_receives_async_observations(make_model):
    sink = RecordingLogSink()
    m = make_model(spec_of(ASYNC_POINT), log_sink=sink)
    out = m.generate(PROMPT, **FIXED)

    assert len(sink.signals) == 2 * 7
    assert [(rid, name) for rid, name, _ in sink.results] == [(out.request_id, "observer")]


# ---------------------------------------------------------------------------
# Custom EngineAdapter backend
# ---------------------------------------------------------------------------


class FakeAdapter(EngineAdapter):
    """Routes one synthetic record per extraction point, then returns fixed text.
    No `last_stop`, so ProbedModel falls back to inline signal histories."""

    def __init__(self):
        self.loaded = None
        self.pending = {}
        self.kwargs_seen = []
        self.unregistered = []

    def load_model(self, model_name_or_path, **kwargs):
        self.loaded = (model_name_or_path, kwargs)

    def register_extraction(self, request_id, extraction_points):
        self.pending[request_id] = list(extraction_points)

    def generate(self, request_id, prompt, generation_kwargs, router):
        self.kwargs_seen.append(generation_kwargs)
        points = self.pending.pop(request_id)
        router.register_request(request_id, points, RequestContext(request_id, {"prompt": prompt}, None))
        try:
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
        finally:
            router.end_request(request_id)
        return f"fake:{prompt}"

    def unregister_extraction(self, request_id):
        self.unregistered.append(request_id)


def test_custom_engine_adapter_backend():
    adapter = FakeAdapter()
    spec = spec_of(point(), point(name="toxicity", probe_type="abort", probe_kind="trajectory", layers=40))
    with ProbedModel.from_pretrained("my-model", backend=adapter, spec=spec, probes=PROBES, device="meta") as m:
        assert m.adapter is adapter
        out = m.generate("hi", max_new_tokens=3, temperature=0.7, repetition_penalty=1.1)

    assert adapter.loaded == ("my-model", {"device": "meta"})
    # Normalised kwargs pass through untranslated, extras included.
    assert adapter.kwargs_seen == [{"max_new_tokens": 3, "temperature": 0.7, "repetition_penalty": 1.1}]
    assert adapter.unregistered == [out.request_id]
    assert out.text == "fake:hi"
    assert out.probe_results["last_prompt"].verdict == 1
    # Abort recovered from the inline point's signal_history.
    assert out.aborted and out.abort_point == "toxicity"
    assert "confidence 0.93" in out.abort_reason


def test_custom_adapter_is_not_closed():
    class ClosableFake(FakeAdapter):
        closed = False

        def close(self):
            self.closed = True

    adapter = ClosableFake()
    ProbedModel.from_pretrained("m", backend=adapter, spec=spec_of(), probes=PROBES).close()
    assert not adapter.closed


# ---------------------------------------------------------------------------
# Generation kwarg translation (HF)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({}, {"max_new_tokens": DEFAULT_MAX_NEW_TOKENS}),
        ({"max_length": 10}, {"max_length": 10}),
        ({"max_new_tokens": 4, "temperature": 0.0, "top_p": 0.9}, {"max_new_tokens": 4, "do_sample": False}),
        (
            {"max_new_tokens": 4, "temperature": 0.7, "top_p": 0.9},
            {"max_new_tokens": 4, "do_sample": True, "temperature": 0.7, "top_p": 0.9},
        ),
        ({"max_new_tokens": 4, "seed": 3, "stop": "\n"}, {"max_new_tokens": 4, "stop_strings": ["\n"]}),
        ({"max_new_tokens": 4, "stop": ["a", "b"]}, {"max_new_tokens": 4, "stop_strings": ["a", "b"]}),
        ({"max_new_tokens": 4, "repetition_penalty": 1.2}, {"max_new_tokens": 4, "repetition_penalty": 1.2}),
    ],
)
def test_hf_kwarg_translation(given, expected):
    assert HFBackend().translate_kwargs(given) == expected


def test_stop_string_truncates_text():
    from undercurrent.model import _cut_at_stop

    assert _cut_at_stop("hello world. more", [".", "zz"]) == "hello world"
    assert _cut_at_stop("hello", ["zz"]) == "hello"
    assert _cut_at_stop("hello", None) == "hello"


def test_seed_makes_sampling_reproducible(make_model):
    m = make_model()
    a = m.generate(PROMPT, max_new_tokens=6, temperature=1.0, seed=7).text
    b = m.generate(PROMPT, max_new_tokens=6, temperature=1.0, seed=7).text
    assert a == b


# ---------------------------------------------------------------------------
# Import weight and the quickstart example
# ---------------------------------------------------------------------------


def test_import_model_does_not_import_torch():
    code = (
        "import sys\n"
        "import undercurrent.model\n"
        "print(','.join(m for m in ('torch', 'transformers', 'vllm') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"heavy modules imported: {out.stdout.strip()}"


QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


def test_quickstart_runs_offline_on_tiny_model(tmp_path):
    model_dir = tmp_path / "tiny-gpt2"
    torch.manual_seed(0)
    make_tiny_gpt2().save_pretrained(model_dir)
    make_tiny_tokenizer().save_pretrained(model_dir)

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    start = time.monotonic()
    out = subprocess.run(
        [sys.executable, str(QUICKSTART), str(model_dir)], capture_output=True, text=True, env=env, timeout=60
    )
    elapsed = time.monotonic() - start

    assert out.returncode == 0, out.stderr
    assert "prompt_norm: verdict=" in out.stdout
    assert elapsed < 30
    assert len(QUICKSTART.read_text(encoding="utf-8").splitlines()) <= 25

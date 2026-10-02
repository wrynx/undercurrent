"""CPU end-to-end test of the real engine path: YAML spec -> HFEngineAdapter
-> Router -> probes -> inline abort, plus async results landing in a
FileLogSink.

Runs a real forward pass on the tiny random GPT-2 from
`tests/adapters/hf/_helpers.py` (2 layers, n_embd=16, vocab 64), so it needs
no GPU and no network (it passes under `HF_HUB_OFFLINE=1`). The probes are the
installed reference probes from `undercurrent.core.examples`, not the
`examples/` demo.

The spec has two points:
  - `gate` (inline, trajectory): residual stream at layer 1 from generated
    token 2 onward, bound to a TrajectoryScoreProbe whose threshold always
    trips, so generation aborts at generated token 2.
  - `observer` (async, trajectory): MLP output at layers 0 and 1 for every
    generated token, results logged to a FileLogSink.

`min_new_tokens == max_new_tokens` keeps the random model's EOS token from
ending generation early, so any early stop is the abort's doing.
"""

import json
import math

import pytest
import torch

from tests.adapters.hf._helpers import make_tiny_gpt2, make_tiny_tokenizer, spy_end_request
from undercurrent.adapters.hf import HFEngineAdapter
from undercurrent.core import ProbeAction
from undercurrent.core.examples import TrajectoryScoreProbe
from undercurrent.router import ProbeFactory, Router
from undercurrent.sinks import FileLogSink
from undercurrent.spec import ExecutionMode, ProbeKind, TensorType, load_yaml_file

SPEC_YAML = """\
version: "1"
extraction_points:
  - name: gate
    layers: 1
    tensor_type: residual_stream
    position: "generated[2:]"
    probe_type: always_abort
    probe_kind: trajectory
    execution_mode: inline
  - name: observer
    layers: [0, 1]
    tensor_type: mlp_out
    position: "generated[*]"
    probe_type: observer
    probe_kind: trajectory
    execution_mode: async
    queue_depth: 64
"""

PROMPT = "tok1 tok2 tok3"
PROMPT_LEN = 3
MAX_NEW_TOKENS = 12
GENERATION_KWARGS = {"max_new_tokens": MAX_NEW_TOKENS, "min_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
ABORT_AT = 2  # the gate's position is generated[2:]
HIDDEN = 16


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text(SPEC_YAML, encoding="utf-8")
    return load_yaml_file(path)


@pytest.fixture
def adapter():
    torch.manual_seed(1234)
    a = HFEngineAdapter()
    a.load_model(make_tiny_gpt2(), tokenizer=make_tiny_tokenizer(), device="cpu")
    return a


@pytest.fixture
def sink(tmp_path):
    return FileLogSink(tmp_path / "observations.ndjson")


@pytest.fixture
def router(sink):
    r = Router(
        {
            # running mean >= -inf always holds: abort on the first activation.
            "always_abort": ProbeFactory(TrajectoryScoreProbe, {"threshold": -math.inf}),
            # never trips: a pure observer.
            "observer": ProbeFactory(TrajectoryScoreProbe, {"threshold": math.inf}),
        }
    )
    r.attach_log_sink(sink)
    yield r
    r.shutdown()


@pytest.fixture
def routed(router):
    """Records every ActivationRecord the adapter hands to `router.route`."""
    records = []
    original = router.route

    def _spy(record):
        records.append(record)
        return original(record)

    router.route = _spy
    return records


def _run(adapter, router, request_id, points):
    captured = spy_end_request(router)
    adapter.register_extraction(request_id, list(points))
    text = adapter.generate(request_id, PROMPT, dict(GENERATION_KWARGS), router)
    return text, captured["results"]


def _sink_results(sink):
    lines = [json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line["kind"] == "result"]


def test_spec_parses_to_inline_and_async_trajectory_points(spec):
    gate, observer = spec.get("gate"), spec.get("observer")
    assert gate.execution_mode == ExecutionMode.INLINE
    assert observer.execution_mode == ExecutionMode.ASYNC
    assert gate.probe_kind == observer.probe_kind == ProbeKind.TRAJECTORY
    assert gate.layers == (1,)
    assert tuple(observer.layers) == (0, 1)


def test_hf_pipeline_end_to_end_with_inline_abort(adapter, router, routed, sink, spec):
    text, results = _run(adapter, router, "req-1", spec)

    # end_request returned a result for every point in the spec.
    assert set(results) == set(spec.names)

    # Activations arrived at the expected layers, tensor types and positions.
    observer_records = [r for r in routed if r.extraction_point_name == "observer"]
    gate_records = [r for r in routed if r.extraction_point_name == "gate"]
    expected_observer = {(layer, PROMPT_LEN + i) for layer in (0, 1) for i in range(ABORT_AT + 1)}
    assert {(r.layer, r.token_pos) for r in observer_records} == expected_observer
    assert len(observer_records) == len(expected_observer)
    assert [(r.layer, r.token_pos) for r in gate_records] == [(1, PROMPT_LEN + ABORT_AT)]
    for record in routed:
        assert record.request_id == "req-1"
        assert record.is_generated is True
        assert tuple(record.tensor.shape) == (HIDDEN,)
        assert torch.isfinite(record.tensor).all()
    assert {r.tensor_type for r in observer_records} == {TensorType.MLP_OUT.value}
    assert {r.tensor_type for r in gate_records} == {TensorType.RESIDUAL_STREAM.value}

    # The inline abort stopped generation early: the generated activations
    # stop at the abort position, and the output is far short of the cap.
    assert len(text.split()) <= ABORT_AT + 2 < MAX_NEW_TOKENS
    gate_result = results["gate"]
    assert gate_result.verdict["count"] == 1
    assert gate_result.verdict["aborted"] is True
    assert [s.action for s in gate_result.signal_history] == [ProbeAction.ABORT]

    # The async point saw everything routed to it before end_request drained it.
    assert results["observer"].verdict["count"] == len(observer_records)
    assert results["observer"].verdict["aborted"] is False

    # Async results land in the FileLogSink; inline results don't.
    logged = _sink_results(sink)
    assert [(r["request_id"], r["extraction_point_name"]) for r in logged] == [("req-1", "observer")]
    assert logged[0]["payload"]["verdict"]["count"] == len(observer_records)


def test_second_request_on_same_router_is_isolated(adapter, router, routed, sink, spec):
    _, first = _run(adapter, router, "req-1", spec)
    first_observed = sum(r.extraction_point_name == "observer" for r in routed)

    # Without the gate, the same router and adapter run to the token cap,
    # and nothing from req-1 (counts, abort state) carries over.
    routed.clear()
    observer_only = [spec.get("observer")]
    _, second = _run(adapter, router, "req-2", observer_only)
    assert set(second) == {"observer"}
    assert {r.request_id for r in routed} == {"req-2"}
    # The last generated token needs no forward pass, so MAX_NEW_TOKENS - 1
    # decode steps, each captured at two layers.
    assert len(routed) == 2 * (MAX_NEW_TOKENS - 1)
    assert second["observer"].verdict["count"] == len(routed)
    assert second["observer"].verdict["aborted"] is False

    # Re-running the aborting request reproduces the first result exactly.
    routed.clear()
    _, third = _run(adapter, router, "req-3", spec)
    assert {r.request_id for r in routed} == {"req-3"}
    assert third["gate"].verdict == first["gate"].verdict
    assert third["observer"].verdict["count"] == first["observer"].verdict["count"] == first_observed
    assert third["gate"] is not first["gate"]

    logged = {r["request_id"]: r["payload"]["verdict"]["count"] for r in _sink_results(sink)}
    assert logged == {"req-1": first_observed, "req-2": 2 * (MAX_NEW_TOKENS - 1), "req-3": first_observed}

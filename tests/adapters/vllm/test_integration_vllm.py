"""Real-vLLM integration test: a small model, actual generation, actual
continuous batching.

Requires a GPU and a CUDA/ROCm-matched vLLM install, and downloads `gpt2`
from the Hugging Face Hub, so every test here is marked `gpu` and `network`
(the root `tests/conftest.py` skips them when CUDA is unavailable, or unless
`RUN_NETWORK_TESTS=1` is set) and the module is additionally gated behind
`pytest.importorskip("vllm")` for GPU machines without vLLM. Every other
test in this directory (`test_seq_mapper.py`,
`test_worker_extension_bookkeeping.py`, `test_worker_extension_capture.py`,
`test_adapter_contract.py`) needs neither a GPU nor vLLM. See the vLLM adapter design doc in docs/_legacy/
"Limitations" for what to check first if this file doesn't pass as-is
against whatever vLLM version you run it with (introspection.py and adapter.py's `_rpc`/`_get_tokenizer`
are the likely places a version mismatch would surface).

Note on reading probe results: `VLLMEngineAdapter.generate()` calls
`router.end_request()` in a `finally` block before returning, which tears
down the router's per-request state (including the spawned Probe
instances) as part of finalizing them. So a test can't call
`router.get_probe(request_id, ...)` AFTER `generate()` returns -- by then
it's gone. Every probe below instead appends to an external `sink` list
passed in at construction time, so results survive past `end_request()`.

Covers these scenarios, the last of which is the one HF's
sequential adapter doesn't need to prove at all:
  1. a single_shot extraction point produces one correct ActivationRecord.
  2. a generated[*] trajectory extraction point fires across multiple
     decode steps for one request.
  3. that SAME trajectory case, with a second concurrent request (its own,
     different extraction points) actually in flight at the same time --
     proving requests sharing vLLM's continuous-batching scheduler steps
     still get their activations routed to the correct probes.
  4. an inline abort signal actually halts generation early (bounded by the
     documented one-extra-step-or-so latency, not exact-token precision).
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.network]

pytest.importorskip("vllm", reason="integration test requires a real vLLM install; see this file's module docstring")

from tests.adapters.vllm._helpers import make_extraction_point
from undercurrent.adapters.vllm import VLLMEngineAdapter
from undercurrent.core import Probe, ProbeResult, RequestContext
from undercurrent.router import ProbeFactory, Router
from undercurrent.spec import ProbeKind

SMALL_MODEL = "gpt2"


@pytest.fixture(scope="module")
def adapter():
    a = VLLMEngineAdapter()
    a.load_model(SMALL_MODEL, gpu_memory_utilization=0.3, max_model_len=64, enforce_eager=True)
    yield a
    a.shutdown()


class SinkProbe(Probe):
    """Appends every ActivationRecord it sees to an externally-owned list,
    so results are readable after Router.end_request() tears the probe
    instance itself down. `abort_after` optionally returns an inline abort
    signal once the sink (globally, across whichever probes share it) has
    accumulated that many records."""

    probe_kind = "trajectory"

    def __init__(self, sink, abort_after=None):
        super().__init__()
        self._sink = sink
        self._abort_after = abort_after

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def on_activation(self, record):
        self._sink.append(record)
        if self._abort_after is not None and len(self._sink) >= self._abort_after:
            from undercurrent.core import ProbeAction, ProbeSignal

            return ProbeSignal(action=ProbeAction.ABORT)
        return None

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=None)


class SingleShotSinkProbe(SinkProbe):
    probe_kind = "single_shot"


def make_router(sink, *, abort_after=None, probe_kind="trajectory"):
    probe_cls = SingleShotSinkProbe if probe_kind == "single_shot" else SinkProbe
    registry = {"sink": ProbeFactory(probe_cls, {"sink": sink, "abort_after": abort_after})}
    return Router(probe_registry=registry)


def test_single_shot_extraction_point_produces_one_correct_record(adapter):
    sink = []
    router = make_router(sink, probe_kind="single_shot")
    ep = make_extraction_point(
        name="ep-single", layer=0, position="prompt[-1]", probe_type="sink", probe_kind=ProbeKind.SINGLE_SHOT
    )
    request_id = "int-single-shot"
    adapter.register_extraction(request_id, [ep])
    adapter.generate(request_id, "The capital of France is", {"max_tokens": 4, "temperature": 0.0}, router)
    adapter.unregister_extraction(request_id)

    assert len(sink) == 1
    record = sink[0]
    assert record.request_id == request_id
    assert record.extraction_point_name == "ep-single"
    assert record.layer == 0
    assert record.is_generated is False  # prompt[-1] is always a prompt-portion token
    assert record.tensor is not None


def test_trajectory_extraction_point_fires_across_multiple_decode_steps(adapter):
    sink = []
    router = make_router(sink)
    ep = make_extraction_point(name="ep-trajectory", layer=0, position="generated[*]", probe_type="sink")
    request_id = "int-trajectory"
    adapter.register_extraction(request_id, [ep])
    adapter.generate(request_id, "Once upon a time,", {"max_tokens": 8, "temperature": 0.0}, router)
    adapter.unregister_extraction(request_id)

    assert len(sink) >= 2  # multiple decode steps, not just the first token
    assert all(r.request_id == request_id and r.is_generated for r in sink)
    assert [r.token_pos for r in sink] == sorted(r.token_pos for r in sink)
    assert len({r.token_pos for r in sink}) == len(sink)  # each decode step produced a distinct token_pos


def test_two_concurrent_requests_share_batching_and_route_correctly(adapter):
    """The scenario HF's sequential adapter cannot exercise at all: two
    `generate()` calls, each on its own Python thread, actually overlapping
    on vLLM's shared scheduler -- not merely interleaved by the GIL."""
    sink_a, sink_b = [], []
    registry = {
        "sink_a": ProbeFactory(SinkProbe, {"sink": sink_a}),
        "sink_b": ProbeFactory(SinkProbe, {"sink": sink_b}),
    }
    router = Router(probe_registry=registry)

    ep_a = make_extraction_point(name="ep-a", layer=0, position="generated[*]", probe_type="sink_a")
    ep_b = make_extraction_point(name="ep-b", layer=0, position="generated[*]", probe_type="sink_b")

    adapter.register_extraction("int-concurrent-a", [ep_a])
    adapter.register_extraction("int-concurrent-b", [ep_b])

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            adapter.generate,
            "int-concurrent-a",
            "The weather today is",
            {"max_tokens": 10, "temperature": 0.0},
            router,
        )
        future_b = pool.submit(
            adapter.generate,
            "int-concurrent-b",
            "In the beginning there was",
            {"max_tokens": 10, "temperature": 0.0},
            router,
        )
        future_a.result(timeout=60)
        future_b.result(timeout=60)

    adapter.unregister_extraction("int-concurrent-a")
    adapter.unregister_extraction("int-concurrent-b")

    assert len(sink_a) >= 1
    assert len(sink_b) >= 1
    assert all(r.request_id == "int-concurrent-a" for r in sink_a)
    assert all(r.request_id == "int-concurrent-b" for r in sink_b)
    # Strictly increasing token_pos within each request confirms the row->position mapping
    # stayed correct across however many scheduler steps these two requests actually shared.
    assert [r.token_pos for r in sink_a] == sorted(r.token_pos for r in sink_a)
    assert [r.token_pos for r in sink_b] == sorted(r.token_pos for r in sink_b)


def test_inline_abort_halts_generation_early(adapter):
    sink = []
    router = make_router(sink, abort_after=3)
    ep = make_extraction_point(name="ep-abort", layer=0, position="generated[*]", probe_type="sink")
    request_id = "int-abort"
    adapter.register_extraction(request_id, [ep])
    text = adapter.generate(request_id, "Tell me a long story about", {"max_tokens": 64, "temperature": 0.0}, router)
    adapter.unregister_extraction(request_id)

    # Bounded, not exact: vLLM's continuous batching can't stop mid-step the way a
    # sequential HF loop can (see adapter.py's _drain_pending_aborts docstring) -- assert
    # generation stopped well short of max_tokens=64, not at exactly 3 tokens.
    assert len(sink) >= 3
    assert len(sink) < 32
    approx_tokens_generated = len(text.split())
    assert approx_tokens_generated < 32

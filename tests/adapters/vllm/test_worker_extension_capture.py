"""End-to-end test of ProbingWorkerExtension's hook-firing -> row-mapping ->
ActivationRecord -> Router.route() -> abort-signal path, without a real
torch or vLLM install.

`_extract_captured_tensor` does `import torch; isinstance(output, torch.Tensor)`.
We install a minimal fake `torch` module into `sys.modules` with a `Tensor`
class our fake activation payloads are instances of -- a standard trick for
exercising import-guarded code without the real (heavy, GPU-matched)
dependency. This does NOT stand in for the real integration test (real
model, real scheduler, real concurrency) in tests/test_integration_vllm.py;
it proves the wiring between "a forward hook fired with some output tensor"
and "the right ActivationRecord reached the right probe" is correct at the
unit level, including the two-concurrent-requests-in-one-step case that is
this adapter's whole reason for existing.

Hooks must fire from INSIDE the (wrapped) `execute_model` call -- the
wrapper clears its per-step scheduler-metadata snapshot in a `finally`
block as soon as the underlying call returns, exactly like the real
`model_runner.execute_model` would once a real forward pass finishes. So
`FiringModelRunner.execute_model` fires whatever layer hooks were armed via
`fire_on_next_step()` itself, simulating what PyTorch's real hook machinery
would do mid-forward-pass, rather than firing them from the test after the
call returns.
"""

import sys
import types

import pytest

from tests.adapters.vllm._helpers import FakeModelRunner, make_extraction_point
from undercurrent.adapters.vllm.worker_extension import ProbingWorkerExtension


class FakeTensor:
    """Stands in for torch.Tensor: supports row indexing (returning another
    FakeTensor, the way real Tensor row-indexing returns another Tensor) and
    the detach()/to() calls `_emit()` makes before handing a row off to the
    Router (never holding a "GPU" reference past that boundary)."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __getitem__(self, row):
        return FakeTensor([self.rows[row]])

    def detach(self):
        return self

    def to(self, device, dtype=None, copy=False):  # mirrors torch.Tensor.to
        return self

    def tolist(self):
        return self.rows[0]

    def __eq__(self, other):
        if isinstance(other, FakeTensor):
            return self.rows == other.rows
        return self.rows == [other]


class FakeInputBatch:
    def __init__(self, req_ids, num_computed_tokens_cpu):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = num_computed_tokens_cpu


class FiringModelRunner(FakeModelRunner):
    """Like FakeModelRunner, but its execute_model stub itself invokes
    whatever layer hooks were armed via fire_on_next_step(), simulating a
    real forward pass -- so hook firing happens inside the wrapped
    execute_model call's scope, the same as it would with real torch/vLLM.
    """

    def __init__(self, num_layers: int = 1) -> None:
        super().__init__(num_layers=num_layers)
        self._armed = {}

        def _execute_model(scheduler_output, *a, **kw):
            self.execute_model_calls += 1
            for layer_idx, output in self._armed.items():
                layer = self.model.model.layers[layer_idx]
                for hook in list(layer.hooks):
                    hook(layer, (), output)
            self._armed = {}
            return "ok"

        self.execute_model = _execute_model

    def fire_on_next_step(self, layer_idx: int, output: FakeTensor) -> None:
        self._armed[layer_idx] = output


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    fake = types.ModuleType("torch")
    fake.Tensor = FakeTensor
    fake.float32 = "float32"
    monkeypatch.setitem(sys.modules, "torch", fake)
    yield fake


def make_request_ctx(request_id):
    from undercurrent.core import RequestContext

    return RequestContext(request_id=request_id, prompt_metadata={}, extraction_point_config=None)


def test_two_concurrent_requests_in_one_step_route_to_correct_probes(router):
    ext = ProbingWorkerExtension()
    runner = FiringModelRunner(num_layers=1)
    ext.model_runner = runner  # type: ignore[attr-defined]
    ext.bind_router(router)

    ep_a = make_extraction_point(name="ep-a", layer=0, position="generated[*]", probe_type="recording")
    ep_b = make_extraction_point(name="ep-b", layer=0, position="generated[*]", probe_type="recording")
    ext.register_extraction("req-a", [ep_a], prompt_len=4)
    ext.register_extraction("req-b", [ep_b], prompt_len=2)
    router.register_request("req-a", [ep_a], make_request_ctx("req-a"))
    router.register_request("req-b", [ep_b], make_request_ctx("req-b"))

    # Both requests share one step: req-a mid-decode (about to produce its
    # 3rd generated token, absolute token_index 6 = 4 prompt + 2 already
    # generated), req-b still finishing a 2-row prefill chunk (rows 0,1 of
    # its own sequence -- both prompt tokens).
    runner.input_batch = FakeInputBatch(req_ids=["req-a", "req-b"], num_computed_tokens_cpu=[6, 0])
    runner.fire_on_next_step(0, FakeTensor(["row0-req-a", "row1-req-b", "row2-req-b"]))
    scheduler_output = types.SimpleNamespace(num_scheduled_tokens={"req-a": 1, "req-b": 2})
    ext.model_runner.execute_model(scheduler_output)

    a_records = router.get_probe("req-a", "ep-a").received
    b_records = router.get_probe("req-b", "ep-b").received

    assert len(a_records) == 1
    assert a_records[0].token_pos == 6
    assert a_records[0].is_generated is True
    assert a_records[0].layer == 0
    assert a_records[0].tensor_type == "residual_stream"
    assert a_records[0].tensor == "row0-req-a"

    # req-b's rows are still prompt tokens; the position selector is
    # generated[*], which never matches prompt rows -- no records yet.
    assert b_records == []

    router.end_request("req-a")
    router.end_request("req-b")


def test_inline_abort_signal_reaches_pending_aborts(router):
    ext = ProbingWorkerExtension()
    runner = FiringModelRunner(num_layers=1)
    ext.model_runner = runner  # type: ignore[attr-defined]
    ext.bind_router(router)

    ep = make_extraction_point(name="ep-abort", layer=0, position="generated[*]", probe_type="recording_abort_after_1")
    ext.register_extraction("req-1", [ep], prompt_len=1)
    router.register_request("req-1", [ep], make_request_ctx("req-1"))

    runner.input_batch = FakeInputBatch(req_ids=["req-1"], num_computed_tokens_cpu=[1])
    runner.fire_on_next_step(0, FakeTensor(["row0"]))
    scheduler_output = types.SimpleNamespace(num_scheduled_tokens={"req-1": 1})
    ext.model_runner.execute_model(scheduler_output)

    assert ext.pop_pending_aborts() == ["req-1"]
    assert ext.pop_pending_aborts() == []  # drained
    router.end_request("req-1")


def test_no_hook_overhead_when_no_extraction_point_needs_the_layer(router):
    """Cheap short-circuit: a hook must not resolve step metadata (or touch
    the router) at all when nothing registered needs this (layer,
    tensor_type)."""
    ext = ProbingWorkerExtension()
    runner = FiringModelRunner(num_layers=2)
    ext.model_runner = runner  # type: ignore[attr-defined]
    ext.bind_router(router)

    ep = make_extraction_point(name="ep-1", layer=1, position="generated[*]", probe_type="recording")
    ext.register_extraction("req-1", [ep], prompt_len=1)
    router.register_request("req-1", [ep], make_request_ctx("req-1"))

    runner.input_batch = FakeInputBatch(req_ids=["req-1"], num_computed_tokens_cpu=[1])
    # Fire layer 0, which nothing is registered for -- must be a no-op.
    runner.fire_on_next_step(0, FakeTensor(["row0"]))
    scheduler_output = types.SimpleNamespace(num_scheduled_tokens={"req-1": 1})
    ext.model_runner.execute_model(scheduler_output)

    assert router.get_probe("req-1", "ep-1").received == []
    router.end_request("req-1")


def test_activation_buffered_instead_of_routed_when_no_router_bound():
    """Cross-process fallback (see worker_extension.py's 'CROSS-PROCESS
    ACTIVATION POLLING'): when bind_router() was never called -- e.g. the
    RPC handing the worker a live Router reference failed because worker
    and driver are in different OS processes -- _emit() must buffer the
    ActivationRecord instead of dropping it, and pop_pending_activations()
    must return it as a plain, msgspec-safe dict (not the ActivationRecord
    dataclass itself)."""
    ext = ProbingWorkerExtension()
    runner = FiringModelRunner(num_layers=1)
    ext.model_runner = runner  # type: ignore[attr-defined]
    # Deliberately no ext.bind_router(...) call -- ext._router stays None.

    ep = make_extraction_point(name="ep-1", layer=0, position="generated[*]", probe_type="recording")
    ext.register_extraction("req-1", [ep], prompt_len=1)

    runner.input_batch = FakeInputBatch(req_ids=["req-1"], num_computed_tokens_cpu=[1])
    runner.fire_on_next_step(0, FakeTensor(["row0"]))
    scheduler_output = types.SimpleNamespace(num_scheduled_tokens={"req-1": 1})
    ext.model_runner.execute_model(scheduler_output)

    pending = ext.pop_pending_activations()
    assert len(pending) == 1
    record = pending[0]
    assert record["request_id"] == "req-1"
    assert record["extraction_point_name"] == "ep-1"
    assert record["layer"] == 0
    assert record["tensor_type"] == "residual_stream"
    assert record["tensor"] == "row0"
    assert record["is_generated"] is True
    assert isinstance(record, dict)

    assert ext.pop_pending_activations() == []  # drained

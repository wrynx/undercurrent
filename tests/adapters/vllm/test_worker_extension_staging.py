"""ProbingWorkerExtension's batched, non-blocking capture of CUDA activations.

For CUDA tensors each hook gathers its matched rows and starts one
non-blocking copy into pinned host memory; records are built after a single
wait per step (see "CAPTURE COST" in worker_extension.py's module docstring).
These tests run on CPU torch: `CudaLike` makes a tensor report `is_cuda`, and
`_stage_rows_cuda` is replaced by a stand-in that does the same gather with a
fake CUDA event that counts waits. The real pinned-memory copy is exercised by
the GPU tests (test_integration_vllm.py, test_residual_stream_gpu.py).
"""

import types

import pytest

torch = pytest.importorskip("torch")

from tests.adapters.vllm._helpers import FakeModelRunner, make_extraction_point  # noqa: E402
from undercurrent.adapters.vllm import worker_extension  # noqa: E402
from undercurrent.adapters.vllm.activation_batch import unpack_activation_batch  # noqa: E402
from undercurrent.adapters.vllm.worker_extension import ProbingWorkerExtension, _gather_rows_float32  # noqa: E402
from undercurrent.core import RequestContext  # noqa: E402

HIDDEN = 4


class CudaLike(torch.Tensor):
    """A CPU tensor that reports `is_cuda`, to take the CUDA staging path."""

    @property
    def is_cuda(self):
        return True


class FakeEvent:
    def __init__(self, log):
        self._log = log

    def synchronize(self):
        self._log.append("sync")


@pytest.fixture
def events(monkeypatch):
    log = []

    def fake_stage(tensor, rows):
        log.append(("stage", tuple(rows)))
        return _gather_rows_float32(tensor.as_subclass(torch.Tensor), rows), FakeEvent(log)

    monkeypatch.setattr(worker_extension, "_stage_rows_cuda", fake_stage)
    return log


class StepRunner(FakeModelRunner):
    """execute_model fires the armed layer hooks with real tensors, then runs `after_forward`."""

    def __init__(self, num_layers=2):
        super().__init__(num_layers=num_layers)
        self.armed = {}
        self.after_forward = None
        self.fail = False

        def _execute_model(scheduler_output, *a, **kw):
            for layer_idx, output in self.armed.items():
                layer = self.model.model.layers[layer_idx]
                for hook in list(layer.hooks):
                    hook(layer, (), output)
            if self.after_forward is not None:
                self.after_forward()
            if self.fail:
                raise RuntimeError("forward failed")
            return "ok"

        self.execute_model = _execute_model


def layer_output(rows, base):
    """A `[rows, HIDDEN]` CUDA-like tensor whose row r is filled with base + r."""
    t = torch.stack([torch.full((HIDDEN,), float(base + r)) for r in range(rows)])
    return t.as_subclass(CudaLike)


def setup(router=None, num_layers=2, eps_by_request=None, num_computed=None):
    ext = ProbingWorkerExtension()
    runner = StepRunner(num_layers=num_layers)
    ext.model_runner = runner  # type: ignore[attr-defined]
    if router is not None:
        ext.bind_router(router)
    for rid, eps in eps_by_request.items():
        ext.register_extraction(rid, eps, prompt_len=2)
        if router is not None:
            router.register_request(rid, eps, RequestContext(rid, {}, None))
    rids = list(eps_by_request)
    runner.input_batch = types.SimpleNamespace(req_ids=rids, num_computed_tokens_cpu=num_computed or [2] * len(rids))
    return ext, runner, types.SimpleNamespace(num_scheduled_tokens=dict.fromkeys(rids, 1))


def test_gather_rows_contiguous_and_scattered_are_float32_copies():
    src = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    for rows in ([1, 2, 3], [0, 2, 5]):
        out = _gather_rows_float32(src, rows)
        assert out.dtype == torch.float32
        assert torch.equal(out, src[rows].float())
    fp32 = torch.zeros(3, 4)
    out = _gather_rows_float32(fp32, [0, 1, 2])
    fp32.add_(1.0)  # a fused-residual layer updating its input in place
    assert torch.equal(out, torch.zeros(3, 4))


def test_in_process_routes_after_the_step_with_one_wait(router, events):
    eps = {
        "req-a": [make_extraction_point(name="a", layer=[0, 1], probe_type="recording")],
        "req-b": [make_extraction_point(name="b", layer=[0, 1], probe_type="recording")],
    }
    ext, runner, sched = setup(router, eps_by_request=eps)
    runner.armed = {0: layer_output(2, 0), 1: layer_output(2, 10)}
    routed_during_forward = []
    runner.after_forward = lambda: routed_during_forward.extend(
        len(router.get_probe(r, n).received) for r, n in [("req-a", "a"), ("req-b", "b")]
    )

    ext.model_runner.execute_model(sched)

    assert routed_during_forward == [0, 0]  # nothing routed while the forward pass runs
    assert events == [("stage", (0, 1)), ("stage", (0, 1)), "sync"]  # one gather per hook, one wait
    a = router.get_probe("req-a", "a").received
    b = router.get_probe("req-b", "b").received
    assert [(r.layer, r.tensor[0].item()) for r in a] == [(0, 0.0), (1, 10.0)]
    assert [(r.layer, r.tensor[0].item()) for r in b] == [(0, 1.0), (1, 11.0)]
    assert all(r.tensor.dtype == torch.float32 and r.tensor.shape == (HIDDEN,) for r in a + b)
    router.end_request("req-a")
    router.end_request("req-b")


def test_inline_abort_is_pending_when_the_step_returns(router, events):
    eps = {"req-1": [make_extraction_point(name="ep", layer=0, probe_type="recording_abort_after_1")]}
    ext, runner, sched = setup(router, num_layers=1, eps_by_request=eps)
    runner.armed = {0: layer_output(1, 0)}

    ext.model_runner.execute_model(sched)

    assert ext.pop_pending_aborts() == ["req-1"]
    router.end_request("req-1")


def test_polling_path_defers_the_wait_to_pop(events):
    eps = {"req-1": [make_extraction_point(name="ep", layer=[0, 1])]}
    ext, runner, sched = setup(eps_by_request=eps)
    runner.armed = {0: layer_output(1, 0), 1: layer_output(1, 10)}

    ext.model_runner.execute_model(sched)
    assert "sync" not in events  # no wait inside the step on the polling path

    pending = unpack_activation_batch(ext.pop_pending_activations())
    assert events.count("sync") == 1
    assert [(r["layer"], r["tensor"][0]) for r in pending] == [(0, 0.0), (1, 10.0)]
    assert unpack_activation_batch(ext.pop_pending_activations()) == []


def test_polling_path_flushes_the_previous_step_at_the_next_step(events):
    eps = {"req-1": [make_extraction_point(name="ep", layer=0)]}
    ext, runner, sched = setup(num_layers=1, eps_by_request=eps)
    runner.armed = {0: layer_output(1, 0)}
    ext.model_runner.execute_model(sched)
    runner.input_batch.num_computed_tokens_cpu = [3]
    runner.armed = {0: layer_output(1, 5)}
    ext.model_runner.execute_model(sched)

    pending = unpack_activation_batch(ext.pop_pending_activations())
    assert [(r["token_pos"], r["tensor"][0]) for r in pending] == [(2, 0.0), (3, 5.0)]


def test_two_extraction_points_on_one_row_get_separate_tensors(router, events):
    eps = {
        "req-1": [
            make_extraction_point(name="x", layer=0, probe_type="recording"),
            make_extraction_point(name="y", layer=0, probe_type="recording"),
        ]
    }
    ext, runner, sched = setup(router, num_layers=1, eps_by_request=eps)
    runner.armed = {0: layer_output(1, 7)}

    ext.model_runner.execute_model(sched)

    (x,) = router.get_probe("req-1", "x").received
    (y,) = router.get_probe("req-1", "y").received
    assert events == [("stage", (0,)), "sync"]  # the shared row is copied from the device once
    x.tensor.zero_()
    assert y.tensor[0].item() == 7.0
    router.end_request("req-1")


def test_failed_step_discards_its_captures(events):
    eps = {"req-1": [make_extraction_point(name="ep", layer=0)]}
    ext, runner, sched = setup(num_layers=1, eps_by_request=eps)
    runner.armed = {0: layer_output(1, 0)}
    runner.fail = True

    with pytest.raises(RuntimeError, match="forward failed"):
        ext.model_runner.execute_model(sched)

    assert unpack_activation_batch(ext.pop_pending_activations()) == []
    assert "sync" not in events


def test_position_matching_runs_once_per_step_not_once_per_layer(router, events, monkeypatch):
    from undercurrent.spec import ExtractionPoint

    calls = []
    original = ExtractionPoint.matches

    def counting(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ExtractionPoint, "matches", counting)
    eps = {
        "req-a": [make_extraction_point(name="a", layer=[0, 1], probe_type="recording")],
        "req-b": [make_extraction_point(name="b", layer=[0, 1], probe_type="recording")],
    }
    ext, runner, sched = setup(router, eps_by_request=eps)
    runner.armed = {0: layer_output(2, 0), 1: layer_output(2, 10)}

    ext.model_runner.execute_model(sched)

    assert sorted(calls) == ["a", "b"]  # one check per (row, point) per step, shared by both layers
    assert [(r.layer, r.tensor[0].item()) for r in router.get_probe("req-a", "a").received] == [(0, 0.0), (1, 10.0)]
    assert [(r.layer, r.tensor[0].item()) for r in router.get_probe("req-b", "b").received] == [(0, 1.0), (1, 11.0)]
    router.end_request("req-a")
    router.end_request("req-b")


def test_no_tensor_extraction_on_a_step_where_nothing_matches(events, monkeypatch):
    extracted = []
    original = worker_extension._extract_captured_tensor

    def counting(output, tensor_type, layer_idx):
        extracted.append(layer_idx)
        return original(output, tensor_type, layer_idx)

    monkeypatch.setattr(worker_extension, "_extract_captured_tensor", counting)
    # prompt[-1] only matches during prefill; row 0 here is a decode token (2 computed, prompt_len 2).
    eps = {"req-1": [make_extraction_point(name="ep", layer=0, position="prompt[-1]", probe_type="recording")]}
    ext, runner, sched = setup(num_layers=1, eps_by_request=eps)
    runner.armed = {0: layer_output(1, 0)}

    ext.model_runner.execute_model(sched)

    assert extracted == []
    assert events == []
    assert unpack_activation_batch(ext.pop_pending_activations()) == []


def test_unregistering_the_last_request_turns_its_hooks_off():
    ext = ProbingWorkerExtension()
    ext.model_runner = StepRunner(num_layers=2)  # type: ignore[attr-defined]
    ext.register_extraction("req-1", [make_extraction_point(name="ep", layer=[0, 1])], prompt_len=2)
    assert ext._capture_keys == {
        (0, worker_extension.TensorType.RESIDUAL_STREAM),
        (1, worker_extension.TensorType.RESIDUAL_STREAM),
    }
    ext.unregister_extraction("req-1")
    assert ext._capture_keys == frozenset()

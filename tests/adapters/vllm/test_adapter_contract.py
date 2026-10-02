"""Unit tests for VLLMEngineAdapter's bookkeeping, ordering contract, and
version-seam fallback logic -- everything that doesn't require a real
engine to be loaded. `load_model()` itself (which needs real torch/vllm) is
exercised only by the gated integration test.
"""

import importlib
import types

import pytest
import torch

import undercurrent.adapters.vllm as vllm_adapter_pkg
from tests.adapters.vllm._helpers import make_extraction_point
from undercurrent.adapters.base import EngineAdapter
from undercurrent.adapters.vllm.adapter import _WORKER_EXTENSION_PATH, VLLMAdapterLimitationError, VLLMEngineAdapter
from undercurrent.adapters.vllm.worker_extension import ProbingWorkerExtension


def test_vllm_engine_adapter_implements_the_full_abc():
    assert issubclass(VLLMEngineAdapter, EngineAdapter)
    # Instantiable -- i.e. no abstractmethod was left unimplemented.
    VLLMEngineAdapter()


def test_package_reexports_the_canonical_engine_adapter_abc():
    assert vllm_adapter_pkg.EngineAdapter is EngineAdapter


def test_worker_extension_path_resolves_to_the_worker_extension_class():
    # vLLM loads the worker extension from this dotted *string*, so a package
    # rename that only rewrites imports would break it silently at engine start.
    module_name, _, class_name = _WORKER_EXTENSION_PATH.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    assert cls is ProbingWorkerExtension


def test_worker_extension_path_resolves_via_vllms_own_loader():
    import_utils = pytest.importorskip("vllm.utils.import_utils")
    assert import_utils.resolve_obj_by_qualname(_WORKER_EXTENSION_PATH) is ProbingWorkerExtension


def test_generate_before_load_model_raises():
    adapter = VLLMEngineAdapter()
    with pytest.raises(VLLMAdapterLimitationError):
        adapter.generate("req-1", "hello", {}, router=object())


def test_generate_without_prior_register_extraction_raises():
    adapter = VLLMEngineAdapter()
    adapter._engine = object()  # pretend load_model() ran
    adapter._loop = object()
    with pytest.raises(VLLMAdapterLimitationError):
        adapter.generate("req-1", "hello", {}, router=object())


def test_register_extraction_rejects_duplicate_pending_request_id():
    adapter = VLLMEngineAdapter()
    eps = [make_extraction_point(name="ep-1")]
    adapter.register_extraction("req-1", eps)
    with pytest.raises(VLLMAdapterLimitationError):
        adapter.register_extraction("req-1", eps)


def test_unregister_extraction_before_load_model_is_a_safe_no_op():
    adapter = VLLMEngineAdapter()
    adapter.register_extraction("req-1", [make_extraction_point(name="ep-1")])
    adapter.unregister_extraction("req-1")  # no engine yet -- must not crash
    assert "req-1" not in adapter._pending_extraction_points


def test_rpc_uses_engine_collective_rpc_when_present():
    adapter = VLLMEngineAdapter()
    calls = []

    class FakeEngine:
        def collective_rpc(self, method, args=()):
            calls.append((method, args))
            return [f"{method}-result"]

    adapter._engine = FakeEngine()
    result = adapter._rpc("register_extraction", args=("req-1",))
    assert result == ["register_extraction-result"]
    assert calls == [("register_extraction", ("req-1",))]


def test_rpc_falls_back_to_nested_model_executor_collective_rpc():
    adapter = VLLMEngineAdapter()
    calls = []

    class FakeExecutor:
        def collective_rpc(self, method, args=()):
            calls.append((method, args))
            return ["ok"]

    class FakeInnerEngine:
        model_executor = FakeExecutor()

    class FakeEngine:
        engine = FakeInnerEngine()

    adapter._engine = FakeEngine()
    result = adapter._rpc("pop_pending_aborts")
    assert result == ["ok"]
    assert calls == [("pop_pending_aborts", ())]


def test_rpc_raises_clearly_when_no_entry_point_found():
    adapter = VLLMEngineAdapter()
    adapter._engine = object()  # neither .collective_rpc nor .engine.model_executor.collective_rpc
    with pytest.raises(VLLMAdapterLimitationError):
        adapter._rpc("register_extraction")


def test_merge_per_worker_results_passes_through_flat_single_worker_list():
    # Some vLLM versions/paths return the worker's own list directly rather
    # than wrapping it in another per-worker list -- must be left alone.
    flat = ["req-1", "req-2"]
    assert VLLMEngineAdapter._merge_per_worker_results(flat) == flat


def test_merge_per_worker_results_passes_through_non_list():
    assert VLLMEngineAdapter._merge_per_worker_results(None) is None
    assert VLLMEngineAdapter._merge_per_worker_results("not-a-list") == "not-a-list"


def test_merge_per_worker_results_unions_disjoint_worker_results():
    # PP-shaped case: different ranks own disjoint layers/requests, so their
    # pending lists never share a key -- every entry from every worker must
    # survive, not just rank 0's.
    per_worker = [["req-a"], ["req-b"], []]
    merged = VLLMEngineAdapter._merge_per_worker_results(per_worker)
    assert set(merged) == {"req-a", "req-b"}


def test_merge_per_worker_results_dedupes_tp_replicated_entries():
    # TP-shaped case: every rank computes an identical full tensor for the
    # same event post-all-reduce, so every rank's hook queues "the same"
    # entry -- taking the union naively would triple-count it.
    per_worker = [["req-1"], ["req-1"], ["req-1"]]
    merged = VLLMEngineAdapter._merge_per_worker_results(per_worker)
    assert merged == ["req-1"]


def test_merge_per_worker_results_uses_key_fn_for_dict_entries():
    per_worker = [
        [{"request_id": "req-1", "v": "rank0-copy"}],
        [{"request_id": "req-1", "v": "rank1-copy"}, {"request_id": "req-2", "v": "rank1-copy"}],
    ]
    merged = VLLMEngineAdapter._merge_per_worker_results(per_worker, key_fn=lambda item: item["request_id"])
    assert [item["request_id"] for item in merged] == ["req-1", "req-2"]


def test_drain_pending_aborts_merges_across_all_workers_not_just_rank_0():
    import asyncio

    adapter = VLLMEngineAdapter()

    async def fake_rpc_async(method, args=()):
        assert method == "pop_pending_aborts"
        return [["req-rank0"], ["req-rank1"]]  # 2 workers, disjoint aborts

    adapter._rpc_async = fake_rpc_async
    aborted = []
    adapter._engine = types.SimpleNamespace(abort=lambda rid: aborted.append(rid))

    asyncio.run(adapter._drain_pending_aborts())

    assert set(aborted) == {"req-rank0", "req-rank1"}


def test_drain_pending_activations_merges_across_all_workers_not_just_rank_0():
    import asyncio

    from undercurrent.core import ProbeAction, ProbeSignal

    adapter = VLLMEngineAdapter()
    adapter._active_requests.update({"req-1", "req-2"})  # in flight, as between register_request and end_request

    def make_record(request_id, layer):
        return {
            "request_id": request_id,
            "extraction_point_name": "ep-1",
            "layer": layer,
            "token_pos": 0,
            "tensor_type": "residual_stream",
            "tensor": [1.0, 2.0],
            "is_generated": True,
        }

    async def fake_rpc_async(method, args=()):
        assert method == "pop_pending_activations"
        # PP-shaped: rank 0 only ever sees layer 0's event, rank 1 only sees layer 16's.
        return [[make_record("req-1", layer=0)], [make_record("req-2", layer=16)]]

    adapter._rpc_async = fake_rpc_async

    routed = []

    class FakeRouter:
        def route(self, record):
            routed.append(record)
            return ProbeSignal(action=ProbeAction.CONTINUE)

    asyncio.run(adapter._drain_pending_activations(FakeRouter()))

    assert {(r.request_id, r.layer) for r in routed} == {("req-1", 0), ("req-2", 16)}


def test_drain_pending_activations_dedupes_tp_replicated_records():
    import asyncio

    from undercurrent.core import ProbeAction, ProbeSignal

    adapter = VLLMEngineAdapter()
    adapter._active_requests.update({"req-1"})  # in flight, as between register_request and end_request
    raw_record = {
        "request_id": "req-1",
        "extraction_point_name": "ep-1",
        "layer": 16,
        "token_pos": 0,
        "tensor_type": "final_norm",
        "tensor": [0.5, 0.5],
        "is_generated": False,
    }

    async def fake_rpc_async(method, args=()):
        # 4-way TP: every rank's hook fires with an identical event.
        return [[raw_record], [raw_record], [raw_record], [raw_record]]

    adapter._rpc_async = fake_rpc_async

    routed = []

    class FakeRouter:
        def route(self, record):
            routed.append(record)
            return ProbeSignal(action=ProbeAction.CONTINUE)

    asyncio.run(adapter._drain_pending_activations(FakeRouter()))

    assert len(routed) == 1  # not 4


def test_drain_pending_activations_sorts_across_pp_ranks_by_token_pos_then_layer():
    """PP-shaped case (docs/_legacy/vllm_tp_pp_activation_extraction_notes.md):
    one extraction point spans layer 5 (rank 0) and layer 20 (rank 1).
    Merging worker-by-worker without re-sorting would hand the probe
    tok1@L5, tok2@L5, tok1@L20, tok2@L20 -- wrong, since tok1's full forward
    pass (through both layers) completes before tok2's even starts. The fix
    must restore tok1@L5, tok1@L20, tok2@L5, tok2@L20."""
    import asyncio

    from undercurrent.core import ProbeAction, ProbeSignal

    adapter = VLLMEngineAdapter()
    adapter._active_requests.update({"req-1"})  # in flight, as between register_request and end_request

    def make_record(token_pos, layer, is_generated=True):
        return {
            "request_id": "req-1",
            "extraction_point_name": "ep-1",
            "layer": layer,
            "token_pos": token_pos,
            "tensor_type": "residual_stream",
            "tensor": [float(token_pos), float(layer)],
            "is_generated": is_generated,
        }

    async def fake_rpc_async(method, args=()):
        # rank 0 (owns layer 5) returns its events first, in its own capture
        # order; rank 1 (owns layer 20) returns its events after.
        return [
            [make_record(0, 5, is_generated=False), make_record(0, 5), make_record(1, 5)],
            [make_record(0, 20, is_generated=False), make_record(0, 20), make_record(1, 20)],
        ]

    adapter._rpc_async = fake_rpc_async

    routed = []

    class FakeRouter:
        def route(self, record):
            routed.append((record.is_generated, record.token_pos, record.layer))
            return ProbeSignal(action=ProbeAction.CONTINUE)

    asyncio.run(adapter._drain_pending_activations(FakeRouter()))

    assert routed == [
        (False, 0, 5),
        (False, 0, 20),
        (True, 0, 5),
        (True, 0, 20),
        (True, 1, 5),
        (True, 1, 20),
    ]


def test_check_executor_topology_accepts_single_worker():
    """collective_rpc returns one result per worker -- a single-element
    list means exactly one worker, regardless of whether model_executor is
    reachable from the driver side (it often isn't -- see this method's
    own docstring)."""
    adapter = VLLMEngineAdapter()
    adapter._rpc = lambda method, args=(): [[]]
    adapter._check_executor_topology(allow_unsupported_executor=False)  # must not raise


def test_check_executor_topology_rejects_multiple_workers_by_default():
    adapter = VLLMEngineAdapter()
    adapter._rpc = lambda method, args=(): [[], [], []]  # 3 workers
    with pytest.raises(VLLMAdapterLimitationError):
        adapter._check_executor_topology(allow_unsupported_executor=False)


def test_check_executor_topology_bypass_flag_permits_multiple_workers():
    adapter = VLLMEngineAdapter()
    adapter._rpc = lambda method, args=(): [[], [], []]  # 3 workers
    adapter._check_executor_topology(allow_unsupported_executor=True)  # must not raise


def test_check_executor_topology_bypass_flag_permits_probe_rpc_failure():
    adapter = VLLMEngineAdapter()

    def raising_rpc(method, args=()):
        raise RuntimeError("no worker topology available")

    adapter._rpc = raising_rpc
    adapter._check_executor_topology(allow_unsupported_executor=True)  # must not raise


def test_check_executor_topology_raises_clearly_when_probe_rpc_fails():
    adapter = VLLMEngineAdapter()

    def raising_rpc(method, args=()):
        raise RuntimeError("no worker topology available")

    adapter._rpc = raising_rpc
    with pytest.raises(VLLMAdapterLimitationError):
        adapter._check_executor_topology(allow_unsupported_executor=False)


def test_ensure_router_bound_success_and_idempotent():
    adapter = VLLMEngineAdapter()
    adapter._rpc = lambda method, args=(): [True]
    router = object()
    adapter._ensure_router_bound(router)
    assert adapter._bound_router is router
    adapter._ensure_router_bound(router)  # same router again: no-op, no error


def test_ensure_router_bound_rejects_a_second_different_router():
    adapter = VLLMEngineAdapter()
    adapter._rpc = lambda method, args=(): [True]
    adapter._ensure_router_bound(object())
    with pytest.raises(VLLMAdapterLimitationError):
        adapter._ensure_router_bound(object())


def test_ensure_router_bound_raises_when_worker_reports_failure():
    adapter = VLLMEngineAdapter()
    adapter._rpc = lambda method, args=(): [False]
    with pytest.raises(VLLMAdapterLimitationError):
        adapter._ensure_router_bound(object())


def test_ensure_router_bound_falls_back_when_rpc_raises():
    """Cross-process fallback: some installed vLLM versions' AsyncLLM (e.g.
    0.28) always run EngineCore in a subprocess, so handing it a live
    Router fails at the RPC-serialization layer (TypeError: ... is not
    serializable) before bind_router() is ever invoked worker-side. This
    must be treated as 'fall back to polling', not a hard failure -- see
    worker_extension.py's 'CROSS-PROCESS ACTIVATION POLLING'."""
    adapter = VLLMEngineAdapter()

    def raising_rpc(method, args=()):
        raise TypeError("Object of type <class 'undercurrent.router.router.Router'> is not serializable")

    adapter._rpc = raising_rpc
    router = object()
    adapter._ensure_router_bound(router)  # must not raise
    assert adapter._bound_router is router
    adapter._ensure_router_bound(router)  # idempotent, still no error


def test_drain_pending_activations_routes_reconstructed_records_in_order():
    import asyncio

    from undercurrent.core import ProbeAction, ProbeSignal

    adapter = VLLMEngineAdapter()
    adapter._active_requests.update({"req-1"})  # in flight, as between register_request and end_request
    raw_records = [
        {
            "request_id": "req-1",
            "extraction_point_name": "ep-1",
            "layer": 0,
            "token_pos": i,
            "tensor_type": "residual_stream",
            "tensor": [float(i)],
            "is_generated": True,
        }
        for i in range(3)
    ]

    async def fake_rpc_async(method, args=()):
        assert method == "pop_pending_activations"
        return raw_records

    adapter._rpc_async = fake_rpc_async

    routed = []

    class FakeRouter:
        def route(self, record):
            routed.append(record)
            return ProbeSignal(action=ProbeAction.CONTINUE)

    asyncio.run(adapter._drain_pending_activations(FakeRouter()))

    assert [r.token_pos for r in routed] == [0, 1, 2]
    assert all(r.request_id == "req-1" and r.extraction_point_name == "ep-1" for r in routed)
    # Records cross the RPC boundary as nested lists; probes get float32 tensors, as on every other path.
    assert all(isinstance(r.tensor, torch.Tensor) and r.tensor.dtype == torch.float32 for r in routed)
    assert [r.tensor.tolist() for r in routed] == [[0.0], [1.0], [2.0]]


def test_drain_pending_activations_aborts_engine_on_abort_signal():
    import asyncio

    from undercurrent.core import ProbeAction, ProbeSignal

    adapter = VLLMEngineAdapter()
    adapter._active_requests.update({"req-abort"})  # in flight, as between register_request and end_request
    raw_record = {
        "request_id": "req-abort",
        "extraction_point_name": "ep-1",
        "layer": 0,
        "token_pos": 0,
        "tensor_type": "residual_stream",
        "tensor": [0.0],
        "is_generated": True,
    }

    async def fake_rpc_async(method, args=()):
        return [raw_record]

    adapter._rpc_async = fake_rpc_async

    class FakeRouter:
        def route(self, record):
            return ProbeSignal(action=ProbeAction.ABORT)

    aborted = []
    adapter._engine = types.SimpleNamespace(abort=lambda rid: aborted.append(rid))

    asyncio.run(adapter._drain_pending_activations(FakeRouter()))

    assert aborted == ["req-abort"]


def _raw_activation(request_id, token_pos=0):
    return {
        "request_id": request_id,
        "extraction_point_name": "ep-1",
        "layer": 0,
        "token_pos": token_pos,
        "tensor_type": "residual_stream",
        "tensor": [1.0],
        "is_generated": True,
    }


def test_drain_pending_activations_drops_records_of_requests_that_already_ended():
    """The worker's buffer is shared by every in-flight request: a drain can see
    late records of a request that already ended (e.g. aborted). Routing them would
    raise RouterError and fail the request that happens to be draining."""
    import asyncio

    adapter = VLLMEngineAdapter()
    adapter._active_requests.add("live")

    async def fake_rpc_async(method, args=()):
        return [_raw_activation("ended"), _raw_activation("live")]

    adapter._rpc_async = fake_rpc_async
    routed = []

    class FakeRouter:
        def route(self, record):
            routed.append(record.request_id)

    asyncio.run(adapter._drain_pending_activations(FakeRouter()))
    assert routed == ["live"]


def test_drain_pending_activations_skips_a_request_that_ends_mid_drain():
    import asyncio

    from undercurrent.router import RouterError

    adapter = VLLMEngineAdapter()
    adapter._active_requests.update({"a", "b"})

    async def fake_rpc_async(method, args=()):
        return [_raw_activation("a"), _raw_activation("b")]

    adapter._rpc_async = fake_rpc_async
    routed = []

    class FakeRouter:
        def route(self, record):
            if record.request_id == "a":
                adapter._active_requests.discard("a")  # generate() for "a" finished meanwhile
                raise RouterError("route(): request_id='a' is not registered or has already ended.")
            routed.append(record.request_id)

    asyncio.run(adapter._drain_pending_activations(FakeRouter()))
    assert routed == ["b"]


def test_drain_pending_activations_still_raises_router_errors_for_live_requests():
    import asyncio

    from undercurrent.router import RouterError

    adapter = VLLMEngineAdapter()
    adapter._active_requests.add("live")

    async def fake_rpc_async(method, args=()):
        return [_raw_activation("live")]

    adapter._rpc_async = fake_rpc_async

    class FakeRouter:
        def route(self, record):
            raise RouterError("route(): record doesn't match its extraction point")

    with pytest.raises(RouterError):
        asyncio.run(adapter._drain_pending_activations(FakeRouter()))

import time

import pytest

from tests.router._helpers import GatedProbe, SlowProbe, make_extraction_point
from undercurrent.core import ActivationRecord
from undercurrent.router import ProbeAction, ProbeFactory, Router, RouterError
from undercurrent.spec import ExecutionMode, ProbeKind, TensorType


def _record_for(point, token_pos=0, request_id="req-1"):
    return ActivationRecord(
        request_id=request_id,
        extraction_point_name=point.name,
        layer=point.layers[0],
        token_pos=token_pos,
        tensor_type=point.tensor_type.value,
        tensor=[float(token_pos)],
        is_generated=True,
    )


def test_inline_dispatch_returns_signal_synchronously(probe_registry, make_request_ctx, make_record):
    router = Router(probe_registry)
    point = make_extraction_point(
        name="ep-1", probe_type="mlp_classifier", probe_kind=ProbeKind.SINGLE_SHOT, execution_mode=ExecutionMode.INLINE
    )
    router.register_request("req-1", [point], make_request_ctx())

    record = make_record(request_id="req-1", extraction_point_name="ep-1", tensor=[1.0, 2.0])
    signal = router.route(record)

    assert signal is not None
    assert signal.action == ProbeAction.CONTINUE


def test_async_dispatch_returns_none_and_does_not_block_caller(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 0.5})
    router = Router(registry)

    point = make_extraction_point(name="ep-1", probe_type="slow", execution_mode=ExecutionMode.ASYNC, queue_depth=4)
    router.register_request("req-1", [point], make_request_ctx())

    start = time.monotonic()
    signal = router.route(_record_for(point))
    elapsed = time.monotonic() - start

    assert signal is None
    assert elapsed < 0.2  # route() returned long before SlowProbe's 0.5s on_activation finishes

    router.end_request("req-1")  # drains, ensuring the background call actually ran
    router.shutdown()


def test_async_queue_overflow_triggers_configured_drop_policy(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["gated"] = ProbeFactory(GatedProbe, {})
    router = Router(registry)  # default_overflow_policy=DROP_OLDEST

    point = make_extraction_point(name="ep-1", probe_type="gated", execution_mode=ExecutionMode.ASYNC, queue_depth=2)
    router.register_request("req-1", [point], make_request_ctx())
    probe = router.get_probe("req-1", "ep-1")

    # Item 1 is picked up by the worker immediately and blocks inside
    # on_activation (the gate); once `started` fires, the queue is
    # guaranteed empty and the worker busy, so items 2-5 land purely in
    # the queue and 3 of them must overflow under queue_depth=2.
    router.route(_record_for(point, token_pos=1))
    assert probe.started.wait(timeout=2)

    for token_pos in (2, 3, 4, 5):
        router.route(_record_for(point, token_pos=token_pos))

    probe.release()
    results = router.end_request("req-1")
    router.shutdown()

    received_positions = [record.token_pos for record in results["ep-1"].verdict]
    # drop_oldest: only the 2 most recently queued (4, 5) survive the
    # backlog, plus item 1 which was already in flight before the backlog
    # ever formed.
    assert received_positions == [1, 4, 5]


def test_route_to_unregistered_request_raises(make_record, probe_registry):
    router = Router(probe_registry)
    with pytest.raises(RouterError, match="is not registered"):
        router.route(make_record(request_id="does-not-exist"))


def test_route_to_unknown_extraction_point_raises(probe_registry, make_request_ctx, make_record):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score")
    router.register_request("req-1", [point], make_request_ctx())

    with pytest.raises(RouterError, match="no extraction point named"):
        router.route(make_record(request_id="req-1", extraction_point_name="not-ep-1"))


def test_route_rejects_record_with_mismatched_layer(probe_registry, make_request_ctx, make_record):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score", layer=5)
    router.register_request("req-1", [point], make_request_ctx())

    bad_record = make_record(request_id="req-1", extraction_point_name="ep-1", layer=99)
    with pytest.raises(RouterError, match="layer"):
        router.route(bad_record)


def test_route_rejects_record_with_mismatched_tensor_type(probe_registry, make_request_ctx, make_record):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score", tensor=TensorType.MLP_OUT)
    router.register_request("req-1", [point], make_request_ctx())

    bad_record = make_record(request_id="req-1", extraction_point_name="ep-1", tensor_type="residual_stream")
    with pytest.raises(RouterError, match="tensor_type"):
        router.route(bad_record)

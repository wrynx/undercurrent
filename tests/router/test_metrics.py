import pytest

from tests.router._helpers import (
    FakeLogSink,
    FakeMetricsSink,
    FlakyProbe,
    GatedProbe,
    SlowProbe,
    make_extraction_point,
    wait_until,
)
from undercurrent.core import ProbeAction
from undercurrent.router import ProbeFactory, Router, RouterError
from undercurrent.spec import ExecutionMode


def _record_for(point, token_pos=0, request_id="req-1"):
    from undercurrent.core import ActivationRecord

    return ActivationRecord(
        request_id=request_id,
        extraction_point_name=point.name,
        layer=point.layers[0],
        token_pos=token_pos,
        tensor_type=point.tensor_type.value,
        tensor=[float(token_pos)],
        is_generated=True,
    )


def test_get_metrics_zeroed_immediately_after_registration(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score", execution_mode=ExecutionMode.ASYNC)
    router.register_request("req-1", [point], make_request_ctx())

    snapshot = router.get_metrics("req-1", "ep-1")

    assert snapshot.queue_depth == 0
    assert snapshot.drop_count == 0
    assert snapshot.activation_count == 0
    assert snapshot.error_count == 0
    assert snapshot.avg_activation_latency_seconds is None

    router.end_request("req-1")
    router.shutdown()


def test_get_metrics_raises_for_inline_extraction_point(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score", execution_mode=ExecutionMode.INLINE)
    router.register_request("req-1", [point], make_request_ctx())

    with pytest.raises(RouterError, match="inline"):
        router.get_metrics("req-1", "ep-1")


def test_get_metrics_raises_for_unregistered_request(probe_registry):
    router = Router(probe_registry)
    with pytest.raises(RouterError, match="is not registered"):
        router.get_metrics("does-not-exist", "ep-1")


def test_get_metrics_raises_after_end_request(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score", execution_mode=ExecutionMode.ASYNC)
    router.register_request("req-1", [point], make_request_ctx())
    router.end_request("req-1")

    with pytest.raises(RouterError):
        router.get_metrics("req-1", "ep-1")


def test_queue_depth_reflects_backlog(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["gated"] = ProbeFactory(GatedProbe, {})
    router = Router(registry)
    point = make_extraction_point(name="ep-1", probe_type="gated", execution_mode=ExecutionMode.ASYNC, queue_depth=8)
    router.register_request("req-1", [point], make_request_ctx())
    probe = router.get_probe("req-1", "ep-1")

    # Item 1 is picked up immediately and blocks the worker on the gate;
    # once `started` fires the queue is guaranteed empty and the worker
    # busy, so items 2-4 land purely in the queue.
    router.route(_record_for(point, token_pos=1))
    assert probe.started.wait(timeout=2)
    for token_pos in (2, 3, 4):
        router.route(_record_for(point, token_pos=token_pos))

    assert router.get_metrics("req-1", "ep-1").queue_depth == 3

    probe.release()
    assert wait_until(lambda: router.get_metrics("req-1", "ep-1").queue_depth == 0)

    router.end_request("req-1")
    router.shutdown()


def test_drop_count_increments_under_drop_oldest(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["gated"] = ProbeFactory(GatedProbe, {})
    router = Router(registry)  # default_overflow_policy=DROP_OLDEST
    point = make_extraction_point(name="ep-1", probe_type="gated", execution_mode=ExecutionMode.ASYNC, queue_depth=2)
    router.register_request("req-1", [point], make_request_ctx())
    probe = router.get_probe("req-1", "ep-1")

    router.route(_record_for(point, token_pos=1))
    assert probe.started.wait(timeout=2)
    for token_pos in (2, 3, 4, 5):  # queue_depth=2 -> 2 fit, 2 evicted
        router.route(_record_for(point, token_pos=token_pos))

    assert router.get_metrics("req-1", "ep-1").drop_count == 2

    probe.release()
    router.end_request("req-1")
    router.shutdown()


def test_drop_count_increments_under_drop_newest(probe_registry, make_request_ctx):
    from undercurrent.router import OverflowPolicy

    registry = dict(probe_registry)
    registry["gated"] = ProbeFactory(GatedProbe, {})
    router = Router(registry, default_overflow_policy=OverflowPolicy.DROP_NEWEST)
    point = make_extraction_point(name="ep-1", probe_type="gated", execution_mode=ExecutionMode.ASYNC, queue_depth=2)
    router.register_request("req-1", [point], make_request_ctx())
    probe = router.get_probe("req-1", "ep-1")

    router.route(_record_for(point, token_pos=1))
    assert probe.started.wait(timeout=2)
    for token_pos in (2, 3, 4, 5):  # queue_depth=2 -> 2 fit, 2 rejected
        router.route(_record_for(point, token_pos=token_pos))

    assert router.get_metrics("req-1", "ep-1").drop_count == 2

    probe.release()
    router.end_request("req-1")
    router.shutdown()


def test_activation_count_and_avg_latency_reflect_processing(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 0.05})
    router = Router(registry)
    point = make_extraction_point(name="ep-1", probe_type="slow", execution_mode=ExecutionMode.ASYNC, queue_depth=16)
    router.register_request("req-1", [point], make_request_ctx())

    n = 4
    for token_pos in range(n):
        router.route(_record_for(point, token_pos=token_pos))

    assert wait_until(lambda: router.get_metrics("req-1", "ep-1").activation_count == n, timeout=5)

    snapshot = router.get_metrics("req-1", "ep-1")
    assert snapshot.avg_activation_latency_seconds is not None
    assert snapshot.avg_activation_latency_seconds >= 0.04  # each on_activation slept ~0.05s
    assert snapshot.error_count == 0

    router.end_request("req-1")
    router.shutdown()


def test_probe_error_is_isolated_recorded_and_forwarded_as_continue_signal(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["flaky"] = ProbeFactory(FlakyProbe, {"raise_on": lambda record: record.token_pos == 1})
    router = Router(registry)
    sink = FakeLogSink()
    router.attach_log_sink(sink)

    point = make_extraction_point(name="ep-1", probe_type="flaky", execution_mode=ExecutionMode.ASYNC, queue_depth=8)
    router.register_request("req-1", [point], make_request_ctx())

    for token_pos in (0, 1, 2):
        router.route(_record_for(point, token_pos=token_pos))

    assert wait_until(lambda: router.get_metrics("req-1", "ep-1").activation_count == 3)

    snapshot = router.get_metrics("req-1", "ep-1")
    assert snapshot.error_count == 1

    # The worker must not have crashed: items before and after the
    # failing one were both still processed.
    probe = router.get_probe("req-1", "ep-1")
    assert [r.token_pos for r in probe.received] == [0, 2]

    # The failure was forwarded through the log sink as a continue signal
    # carrying the error, not silently dropped or raised out of the pool.
    assert len(sink.signals) == 1
    _, _, signal = sink.signals[0]
    assert signal.action == ProbeAction.CONTINUE
    assert signal.metadata["router_error"] is True
    assert signal.metadata["error_type"] == "ValueError"

    router.end_request("req-1")
    router.shutdown()


def test_probe_error_in_one_binding_does_not_affect_other_requests_sharing_the_pool(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["flaky"] = ProbeFactory(FlakyProbe, {"raise_on": lambda record: True})  # always raises
    router = Router(registry)

    flaky_point = make_extraction_point(
        name="ep-1", probe_type="flaky", execution_mode=ExecutionMode.ASYNC, queue_depth=8
    )
    healthy_point = make_extraction_point(
        name="ep-1", probe_type="trajectory_score", execution_mode=ExecutionMode.ASYNC, queue_depth=8
    )
    router.register_request("req-flaky", [flaky_point], make_request_ctx(request_id="req-flaky"))
    router.register_request("req-healthy", [healthy_point], make_request_ctx(request_id="req-healthy"))

    for token_pos in range(5):
        router.route(_record_for(flaky_point, token_pos=token_pos, request_id="req-flaky"))
        router.route(_record_for(healthy_point, token_pos=token_pos, request_id="req-healthy"))

    assert wait_until(lambda: router.get_metrics("req-flaky", "ep-1").activation_count == 5)
    assert wait_until(lambda: router.get_metrics("req-healthy", "ep-1").activation_count == 5)

    assert router.get_metrics("req-flaky", "ep-1").error_count == 5
    assert router.get_metrics("req-healthy", "ep-1").error_count == 0

    results = {
        "req-flaky": router.end_request("req-flaky"),
        "req-healthy": router.end_request("req-healthy"),
    }
    router.shutdown()

    assert results["req-healthy"]["ep-1"].verdict["count"] == 5


def test_attach_metrics_sink_forwards_all_event_types(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["gated"] = ProbeFactory(GatedProbe, {})
    registry["flaky"] = ProbeFactory(FlakyProbe, {"raise_on": lambda record: record.token_pos == 1})
    router = Router(registry)
    sink = FakeMetricsSink()
    router.attach_metrics_sink(sink)

    # Deterministic drop: gate the worker on item 1, then overflow the
    # queue_depth=2 backlog with items 2-5 (same pattern as
    # test_drop_count_increments_under_drop_oldest).
    gated_point = make_extraction_point(
        name="ep-1", probe_type="gated", execution_mode=ExecutionMode.ASYNC, queue_depth=2
    )
    router.register_request("req-gated", [gated_point], make_request_ctx(request_id="req-gated"))
    probe = router.get_probe("req-gated", "ep-1")
    router.route(_record_for(gated_point, token_pos=1, request_id="req-gated"))
    assert probe.started.wait(timeout=2)
    for token_pos in (2, 3, 4, 5):
        router.route(_record_for(gated_point, token_pos=token_pos, request_id="req-gated"))
    probe.release()

    # Deterministic error: FlakyProbe raises on token_pos=1.
    flaky_point = make_extraction_point(
        name="ep-1", probe_type="flaky", execution_mode=ExecutionMode.ASYNC, queue_depth=8
    )
    router.register_request("req-flaky", [flaky_point], make_request_ctx(request_id="req-flaky"))
    for token_pos in (0, 1, 2):
        router.route(_record_for(flaky_point, token_pos=token_pos, request_id="req-flaky"))

    assert wait_until(lambda: len(sink.drops) == 2)
    assert wait_until(lambda: len(sink.errors) == 1)

    router.end_request("req-gated")
    router.end_request("req-flaky")
    router.shutdown()

    assert sink.drops == [("req-gated", "ep-1")] * 2
    assert sink.errors == [("req-flaky", "ep-1")]
    assert len(sink.activations) >= 1
    assert len(sink.queue_depths) >= 1


def test_raising_external_metrics_sink_does_not_affect_dispatch(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    router = Router(registry)
    sink = FakeMetricsSink(raise_on="record_activation")
    router.attach_metrics_sink(sink)

    point = make_extraction_point(
        name="ep-1", probe_type="trajectory_score", execution_mode=ExecutionMode.ASYNC, queue_depth=8
    )
    router.register_request("req-1", [point], make_request_ctx())

    for token_pos in range(3):
        router.route(_record_for(point, token_pos=token_pos))

    # Despite the external sink raising on every record_activation call,
    # the router's own registry (fed before forwarding) keeps working and
    # the worker keeps processing subsequent items.
    assert wait_until(lambda: router.get_metrics("req-1", "ep-1").activation_count == 3)

    results = router.end_request("req-1")
    router.shutdown()
    assert results["ep-1"].verdict["count"] == 3

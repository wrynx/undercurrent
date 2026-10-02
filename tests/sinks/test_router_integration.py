import time

from tests.sinks._helpers import RecordingSink, make_extraction_point, make_record
from undercurrent.core import ProbeAction
from undercurrent.sinks import FileLogSink
from undercurrent.spec import ExecutionMode


def test_attach_log_sink_forwards_signals_from_async_trajectory_probe(router, make_request_ctx):
    sink = RecordingSink()
    router.attach_log_sink(sink)

    point = make_extraction_point(name="ep-1", probe_type="emitting", execution_mode=ExecutionMode.ASYNC)
    router.register_request("req-1", [point], make_request_ctx())

    for token_pos in range(3):
        router.route(make_record(point, token_pos=token_pos))

    router.end_request("req-1")

    assert len(sink.signals) == 3
    for (request_id, extraction_point_name, signal), token_pos in zip(sink.signals, range(3)):
        assert request_id == "req-1"
        assert extraction_point_name == "ep-1"
        assert signal.action == ProbeAction.CONTINUE
        assert signal.metadata == {"token_pos": token_pos}


def test_attach_log_sink_forwards_result_once_end_request_finalizes(router, make_request_ctx):
    sink = RecordingSink()
    router.attach_log_sink(sink)

    point = make_extraction_point(name="ep-1", probe_type="emitting", execution_mode=ExecutionMode.ASYNC)
    router.register_request("req-1", [point], make_request_ctx())
    router.route(make_record(point, token_pos=0))
    router.route(make_record(point, token_pos=1))

    results = router.end_request("req-1")

    assert len(sink.results) == 1
    request_id, extraction_point_name, result = sink.results[0]
    assert request_id == "req-1"
    assert extraction_point_name == "ep-1"
    assert result.verdict == {"count": 2}
    assert result is results["ep-1"]


def test_inline_extraction_point_does_not_forward_to_log_sink(router, make_request_ctx):
    sink = RecordingSink()
    router.attach_log_sink(sink)

    point = make_extraction_point(name="ep-1", probe_type="emitting", execution_mode=ExecutionMode.INLINE)
    router.register_request("req-1", [point], make_request_ctx())
    router.route(make_record(point, token_pos=0))
    router.end_request("req-1")

    assert sink.signals == []
    assert sink.results == []


def test_attach_log_sink_can_be_called_after_register_request(router, make_request_ctx):
    point = make_extraction_point(name="ep-1", probe_type="emitting", execution_mode=ExecutionMode.ASYNC)
    router.register_request("req-1", [point], make_request_ctx())

    sink = RecordingSink()
    router.attach_log_sink(sink)  # attached after registration -- must still take effect

    router.route(make_record(point, token_pos=0))
    router.end_request("req-1")

    assert len(sink.signals) == 1


def test_file_log_sink_end_to_end_through_router(router, make_request_ctx, tmp_path):
    sink = FileLogSink(tmp_path / "observations.ndjson")
    router.attach_log_sink(sink)

    point = make_extraction_point(name="ep-1", probe_type="emitting", execution_mode=ExecutionMode.ASYNC)
    router.register_request("req-1", [point], make_request_ctx())
    router.route(make_record(point, token_pos=0))
    router.route(make_record(point, token_pos=1))
    router.end_request("req-1")

    import json

    lines = sink.path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    kinds = [r["kind"] for r in records]
    assert kinds.count("signal") == 2
    assert kinds.count("result") == 1


def test_route_dispatch_latency_stays_low_with_slow_log_sink_attached(probe_registry, make_request_ctx):
    from undercurrent.router import Router

    router = Router(probe_registry)
    try:
        slow_sink = RecordingSink(delay=0.5)
        router.attach_log_sink(slow_sink)

        point = make_extraction_point(name="ep-1", probe_type="emitting", execution_mode=ExecutionMode.ASYNC)
        router.register_request("req-1", [point], make_request_ctx())

        start = time.monotonic()
        router.route(make_record(point, token_pos=0))
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # route() must return long before the sink's 0.5s write_signal finishes

        router.end_request("req-1")  # drains, proving the slow write actually happened
        assert len(slow_sink.signals) == 1
    finally:
        router.shutdown()


def test_route_dispatch_latency_stays_low_with_failing_log_sink_attached(probe_registry, make_request_ctx):
    from undercurrent.router import Router

    router = Router(probe_registry)
    try:
        failing_sink = RecordingSink(raise_on_signal=True)
        router.attach_log_sink(failing_sink)

        point = make_extraction_point(name="ep-1", probe_type="emitting", execution_mode=ExecutionMode.ASYNC)
        router.register_request("req-1", [point], make_request_ctx())

        start = time.monotonic()
        router.route(make_record(point, token_pos=0))
        elapsed = time.monotonic() - start

        assert elapsed < 0.1

        # A raising sink must not crash the worker or end_request -- the
        # probe's own result is still finalized correctly.
        results = router.end_request("req-1")
        assert results["ep-1"].verdict == {"count": 1}
    finally:
        router.shutdown()


def test_dispatch_latency_stays_low_across_many_records_with_slow_sink(probe_registry, make_request_ctx):
    """Repeated route() calls under a slow sink shouldn't individually
    slow down -- each just enqueues onto the binding's own bounded queue,
    independent of how far behind the worker/sink have fallen (subject to
    the queue's overflow policy, exercised elsewhere)."""
    from undercurrent.router import Router

    router = Router(probe_registry, default_queue_depth=64)
    try:
        slow_sink = RecordingSink(delay=0.05)
        router.attach_log_sink(slow_sink)

        point = make_extraction_point(
            name="ep-1", probe_type="emitting", execution_mode=ExecutionMode.ASYNC, queue_depth=64
        )
        router.register_request("req-1", [point], make_request_ctx())

        start = time.monotonic()
        for token_pos in range(20):
            router.route(make_record(point, token_pos=token_pos))
        elapsed = time.monotonic() - start

        assert elapsed < 0.2  # 20 enqueues, not 20 * 0.05s of sink latency

        router.end_request("req-1")
    finally:
        router.shutdown()

import pytest

from tests.router._helpers import make_extraction_point
from undercurrent.core import ProbeResult
from undercurrent.router import ProbeFactory, Router, RouterError
from undercurrent.spec import ExecutionMode, ProbeKind


def test_end_request_returns_result_per_extraction_point(probe_registry, make_request_ctx, make_record):
    router = Router(probe_registry)
    points = [
        make_extraction_point(name="ep-a", probe_type="trajectory_score"),
        make_extraction_point(name="ep-b", probe_type="mlp_classifier", probe_kind=ProbeKind.SINGLE_SHOT),
    ]
    router.register_request("req-1", points, make_request_ctx())

    results = router.end_request("req-1")

    assert set(results.keys()) == {"ep-a", "ep-b"}
    assert all(isinstance(r, ProbeResult) for r in results.values())


def test_end_request_tears_down_state(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score")
    router.register_request("req-1", [point], make_request_ctx())

    router.end_request("req-1")

    with pytest.raises(RouterError):
        router.get_probe("req-1", "ep-1")
    with pytest.raises(RouterError):
        router.end_request("req-1")  # already ended


def test_end_request_drains_async_queue_for_trajectory_probe_before_finalizing(
    probe_registry, make_request_ctx, make_record
):
    router = Router(probe_registry)
    point = make_extraction_point(
        name="ep-1", probe_type="trajectory_score", execution_mode=ExecutionMode.ASYNC, queue_depth=16
    )
    router.register_request("req-1", [point], make_request_ctx())

    # Fire a burst of activations and immediately end the request, without
    # waiting for the async worker to have processed anything -- simulates
    # generation being aborted mid-stream right after the last token.
    for value in (0.1, 0.2, 0.3, 0.4):
        router.route(make_record(request_id="req-1", extraction_point_name="ep-1", tensor=[value, value]))

    results = router.end_request("req-1")
    router.shutdown()

    verdict = results["ep-1"].verdict
    # all 4 activations must have been drained and folded into the running
    # mean before on_end was called, not silently dropped by an early
    # teardown.
    assert verdict["count"] == 4
    assert verdict["final_mean"] == pytest.approx((0.1 + 0.2 + 0.3 + 0.4) / 4)


def test_end_request_finalizes_correctly_when_threshold_triggers_abort_mid_stream(
    probe_registry, make_request_ctx, make_record
):
    registry = dict(probe_registry)
    registry["trajectory_score"] = ProbeFactory(registry["trajectory_score"].probe_cls, {"threshold": 0.25})
    router = Router(registry)

    point = make_extraction_point(
        name="ep-1", probe_type="trajectory_score", execution_mode=ExecutionMode.ASYNC, queue_depth=16
    )
    router.register_request("req-1", [point], make_request_ctx())

    # Running mean crosses 0.25 partway through -- simulating an adapter
    # that would abort generation once it sees that signal, but a few more
    # tokens were already in flight before the abort took effect.
    for value in (0.1, 0.5, 0.9, 0.9):
        router.route(make_record(request_id="req-1", extraction_point_name="ep-1", tensor=[value, value]))

    results = router.end_request("req-1")
    router.shutdown()

    verdict = results["ep-1"].verdict
    assert verdict["aborted"] is True
    assert verdict["count"] == 4  # every queued activation still got folded in before finalizing


def test_end_request_works_for_inline_extraction_point_with_zero_activations(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score", execution_mode=ExecutionMode.INLINE)
    router.register_request("req-1", [point], make_request_ctx())

    results = router.end_request("req-1")

    assert results["ep-1"].verdict == {"final_mean": 0.0, "count": 0, "aborted": False}

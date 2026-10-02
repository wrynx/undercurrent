"""Requirement #3: no data structure in Router should ever key or aggregate
across request_ids. Two concurrent requests -- even reusing the exact same
extraction_point_name -- must never see each other's probe state or
activation records.
"""

import threading

from tests.router._helpers import make_extraction_point
from undercurrent.router import Router
from undercurrent.spec import ExecutionMode


def test_two_concurrent_requests_with_same_extraction_point_name_stay_isolated(
    probe_registry, make_request_ctx, make_record
):
    router = Router(probe_registry)
    # deliberately the *same* extraction_point_name across two requests
    point = make_extraction_point(
        name="shared-name", probe_type="trajectory_score", execution_mode=ExecutionMode.INLINE
    )

    router.register_request("req-A", [point], make_request_ctx(request_id="req-A"))
    router.register_request("req-B", [point], make_request_ctx(request_id="req-B"))

    probe_a = router.get_probe("req-A", "shared-name")
    probe_b = router.get_probe("req-B", "shared-name")
    assert probe_a is not probe_b

    router.route(make_record(request_id="req-A", extraction_point_name="shared-name", tensor=[10.0, 10.0]))
    router.route(make_record(request_id="req-A", extraction_point_name="shared-name", tensor=[10.0, 10.0]))
    router.route(make_record(request_id="req-B", extraction_point_name="shared-name", tensor=[1.0, 1.0]))

    assert probe_a._count == 2
    assert probe_a.running_mean == 10.0
    assert probe_b._count == 1
    assert probe_b.running_mean == 1.0

    results = {
        "req-A": router.end_request("req-A"),
        "req-B": router.end_request("req-B"),
    }

    assert results["req-A"]["shared-name"].verdict["final_mean"] == 10.0
    assert results["req-A"]["shared-name"].verdict["count"] == 2
    assert results["req-B"]["shared-name"].verdict["final_mean"] == 1.0
    assert results["req-B"]["shared-name"].verdict["count"] == 1

    # Router-internal structures must never key/aggregate across
    # request_ids: every dict keyed by request_id must be independently
    # empty-able without touching the other.
    assert "req-A" not in router._requests
    assert "req-B" not in router._requests


def test_concurrent_route_calls_from_different_threads_stay_isolated(probe_registry, make_request_ctx, make_record):
    router = Router(probe_registry)
    point = make_extraction_point(
        name="shared-name", probe_type="trajectory_score", execution_mode=ExecutionMode.INLINE
    )

    request_ids = [f"req-{i}" for i in range(8)]
    for rid in request_ids:
        router.register_request(rid, [point], make_request_ctx(request_id=rid))

    def _drive(request_id, value):
        for _ in range(20):
            router.route(make_record(request_id=request_id, extraction_point_name="shared-name", tensor=[value, value]))

    threads = [threading.Thread(target=_drive, args=(rid, float(i))) for i, rid in enumerate(request_ids)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    for i, rid in enumerate(request_ids):
        probe = router.get_probe(rid, "shared-name")
        assert probe._count == 20
        assert probe.running_mean == float(i)  # no cross-contamination from other requests' values

    for rid in request_ids:
        router.end_request(rid)

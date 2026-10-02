"""Requirement #4: router.shutdown() must drain-or-cancel every in-flight
async binding cleanly, configurable between wait-for-drain (wait=True,
default) and cancel-immediately (wait=False)."""

import threading
import time

from tests.router._helpers import GatedProbe, make_extraction_point
from undercurrent.router import ProbeFactory, Router, RouterError
from undercurrent.spec import ExecutionMode


def _record(point, token_pos, request_id="req-1"):
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


def test_shutdown_wait_true_drains_full_backlog_before_returning(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["gated"] = ProbeFactory(GatedProbe, {})
    router = Router(registry)
    point = make_extraction_point(name="ep-1", probe_type="gated", execution_mode=ExecutionMode.ASYNC, queue_depth=8)
    router.register_request("req-1", [point], make_request_ctx())
    probe = router.get_probe("req-1", "ep-1")

    router.route(_record(point, token_pos=0))
    assert probe.started.wait(timeout=2)  # worker now gated on item 0
    for token_pos in (1, 2, 3):
        router.route(_record(point, token_pos=token_pos))

    def _release_soon():
        time.sleep(0.1)
        probe.release()

    threading.Thread(target=_release_soon).start()

    # shutdown(wait=True) must block until the whole backlog (items 1-3,
    # queued behind the gate) has actually been processed, not just until
    # the gate is released.
    router.shutdown(wait=True)

    assert [r.token_pos for r in probe.received] == [0, 1, 2, 3]


def test_shutdown_wait_false_discards_backlog_but_lets_in_flight_item_finish(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["gated"] = ProbeFactory(GatedProbe, {})
    router = Router(registry)
    point = make_extraction_point(name="ep-1", probe_type="gated", execution_mode=ExecutionMode.ASYNC, queue_depth=8)
    router.register_request("req-1", [point], make_request_ctx())
    probe = router.get_probe("req-1", "ep-1")

    router.route(_record(point, token_pos=0))
    assert probe.started.wait(timeout=2)  # worker now gated inside on_activation(item 0)
    for token_pos in (1, 2, 3):
        router.route(_record(point, token_pos=token_pos))

    # cancel-immediately: returns promptly without waiting for item 0's
    # gate, discarding the queued backlog (1, 2, 3) rather than
    # processing it.
    start = time.monotonic()
    router.shutdown(wait=False)
    assert time.monotonic() - start < 0.5

    # Item 0 was already in flight -- per the documented limitation, a
    # plain Python thread can't be preempted mid-call, so it still runs
    # to completion once released.
    probe.release()
    assert probe.started.wait(timeout=2)
    time.sleep(0.1)  # let the worker thread finish appending
    assert [r.token_pos for r in probe.received] == [0]  # 1, 2, 3 were discarded, not processed


def test_shutdown_is_terminal_route_and_end_request_fail_afterward(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score", execution_mode=ExecutionMode.ASYNC)
    router.register_request("req-1", [point], make_request_ctx())

    router.shutdown()

    try:
        router.route(_record(point, token_pos=0))
        assert False, "expected RouterError"
    except RouterError:
        pass
    try:
        router.end_request("req-1")
        assert False, "expected RouterError"
    except RouterError:
        pass


def test_shutdown_with_no_registered_requests_is_a_no_op(probe_registry):
    router = Router(probe_registry)
    router.shutdown()  # must not raise


def test_shutdown_drains_multiple_independent_requests_without_cross_contamination(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["gated"] = ProbeFactory(GatedProbe, {})
    router = Router(registry)

    point = make_extraction_point(name="ep-1", probe_type="gated", execution_mode=ExecutionMode.ASYNC, queue_depth=4)
    request_ids = [f"req-{i}" for i in range(4)]
    probes = {}
    for rid in request_ids:
        router.register_request(rid, [point], make_request_ctx(request_id=rid))
        probe = router.get_probe(rid, "ep-1")
        probes[rid] = probe
        router.route(_record(point, token_pos=0, request_id=rid))
        assert probe.started.wait(timeout=2)
        router.route(_record(point, token_pos=1, request_id=rid))

    for probe in probes.values():
        probe.release()

    router.shutdown(wait=True)

    for rid in request_ids:
        assert [r.token_pos for r in probes[rid].received] == [0, 1]

"""Requirement #5: 50+ concurrent requests, each streaming a
generated[*]-shaped trajectory of 20-50 tokens through async dispatch, with
one probe deliberately slow. Confirms:
  - no cross-request state leakage (the isolation pattern from
    test_isolation.py, exercised through async dispatch this time)
  - queue depth metrics reflect the slow probe's backlog accurately
  - the other (fast) requests' throughput isn't catastrophically degraded

`worker_pool_size` is set generously above the concurrent-request count
here -- see `router.default_worker_pool_size`'s docstring on why this
matters: each async binding pins one worker thread for its whole request
lifetime, so a pool smaller than the number of concurrently-live async
bindings would starve the excess ones entirely rather than merely slow
them down. That's a real, documented tradeoff of this router's design;
this test is about exercising it correctly-configured, not exposing it.
"""

import random
import threading
import time

from tests.router._helpers import SlowProbe, make_extraction_point, wait_until
from undercurrent.core import ActivationRecord
from undercurrent.router import ProbeFactory, Router
from undercurrent.spec import ExecutionMode

NUM_FAST_REQUESTS = 49
NUM_SLOW_REQUESTS = 1
TOKENS_PER_REQUEST_RANGE = (20, 50)
SLOW_PROBE_DELAY = 0.05


def _record(point, token_pos, request_id, value):
    return ActivationRecord(
        request_id=request_id,
        extraction_point_name=point.name,
        layer=point.layers[0],
        token_pos=token_pos,
        tensor_type=point.tensor_type.value,
        tensor=[value, value],
        is_generated=True,
    )


def test_load_50_concurrent_requests_one_slow_probe_no_leakage_and_healthy_throughput(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": SLOW_PROBE_DELAY})
    router = Router(
        registry,
        worker_pool_size=64,  # >= total concurrently-live bindings below -- see module docstring
        default_queue_depth=64,  # generous: this test measures backlog/throughput, not overflow behavior
    )

    fast_ids = [f"fast-{i}" for i in range(NUM_FAST_REQUESTS)]
    slow_ids = [f"slow-{i}" for i in range(NUM_SLOW_REQUESTS)]
    all_ids = fast_ids + slow_ids
    random.Random(0).shuffle(all_ids)  # registration order shouldn't matter

    points = {}
    token_counts = {}
    rng = random.Random(1)
    for rid in all_ids:
        is_slow = rid in slow_ids
        probe_type = "slow" if is_slow else "trajectory_score"
        point = make_extraction_point(
            name="ep-1", probe_type=probe_type, execution_mode=ExecutionMode.ASYNC, queue_depth=64
        )
        points[rid] = point
        token_counts[rid] = rng.randint(*TOKENS_PER_REQUEST_RANGE)
        router.register_request(rid, [point], make_request_ctx(request_id=rid))

    fast_durations = {}
    slow_started = threading.Event()

    def _drive(rid, value):
        point = points[rid]
        start = time.monotonic()
        for token_pos in range(token_counts[rid]):
            router.route(_record(point, token_pos, rid, value))
            if rid in slow_ids:
                slow_started.set()
        duration = time.monotonic() - start
        if rid in fast_ids:
            fast_durations[rid] = duration

    threads = [threading.Thread(target=_drive, args=(rid, float(i))) for i, rid in enumerate(all_ids)]
    overall_start = time.monotonic()
    for t in threads:
        t.start()

    # While the slow request's tokens are still being fed in and its
    # worker is backed up processing them at SLOW_PROBE_DELAY each, its
    # queue depth should show real backlog -- this is the "queue depth
    # metrics reflect the slow probe accurately" assertion. Poll rather
    # than sleep a fixed amount: thread start order isn't guaranteed.
    assert wait_until(lambda: slow_started.is_set(), timeout=5)
    assert wait_until(lambda: router.get_metrics(slow_ids[0], "ep-1").queue_depth > 0, timeout=5)

    for t in threads:
        t.join(timeout=30)
    dispatch_elapsed = time.monotonic() - overall_start

    # Throughput: route() for the 49 fast requests must not have been
    # dragged down by the one slow binding -- each is independently
    # queued and processed by its own worker thread (see module
    # docstring: one dedicated worker per binding), so fast dispatch
    # should complete in a small fraction of a second regardless of how
    # long the slow binding's backlog takes to drain.
    assert fast_durations, "no fast request timings collected"
    assert max(fast_durations.values()) < 1.0
    assert dispatch_elapsed < 2.0  # route() calls themselves never block on the slow binding

    results = {rid: router.end_request(rid) for rid in all_ids}
    router.shutdown()

    # No cross-request state leakage: every request's verdict reflects
    # exactly its own token count and its own tensor value (the
    # isolation pattern from test_isolation.py), despite every binding
    # sharing one pool and racing concurrently.
    for i, rid in enumerate(all_ids):
        verdict = results[rid]["ep-1"].verdict
        expected_value = float(i)
        if rid in fast_ids:
            assert verdict["count"] == token_counts[rid]
            assert verdict["final_mean"] == expected_value
        else:
            assert len(verdict) == token_counts[rid]  # SlowProbe's verdict is the list of received records
            assert all(r.tensor == [expected_value, expected_value] for r in verdict)
            assert [r.token_pos for r in verdict] == list(range(token_counts[rid]))

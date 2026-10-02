"""Requirement #1: the overflow policy (drop_oldest | drop_newest | block)
must hold under *concurrent* producers -- many threads calling route() at
once, racing each other and the single consumer worker -- not just a single
producer as in test_dispatch.py's deterministic gated tests. These tests
confirm the accounting invariant that must hold regardless of interleaving:
every record either ends up received by the probe or counted as a drop,
never both, never neither (no silent duplication or loss outside the
documented drop behavior), and that BLOCK genuinely applies backpressure to
the calling (producer) thread rather than letting the queue grow unbounded.
"""

import itertools
import threading
import time

from tests.router._helpers import GatedProbe, SlowProbe, make_extraction_point
from undercurrent.core import ActivationRecord
from undercurrent.router import OverflowPolicy, ProbeFactory, Router
from undercurrent.spec import ExecutionMode


def _record(point, token_pos, request_id="req-1"):
    return ActivationRecord(
        request_id=request_id,
        extraction_point_name=point.name,
        layer=point.layers[0],
        token_pos=token_pos,
        tensor_type=point.tensor_type.value,
        tensor=[float(token_pos)],
        is_generated=True,
    )


def _run_concurrent_producers(router, point, num_producers=8, per_producer=200):
    """Fire `num_producers` threads, each pushing `per_producer` records
    with a globally-unique token_pos (via a shared counter), as fast as
    possible against `router`. Returns the full set of token_pos values
    that were actually submitted."""
    counter = itertools.count()
    submitted = []
    submitted_lock = threading.Lock()
    barrier = threading.Barrier(num_producers)

    def _produce():
        barrier.wait()  # maximize actual concurrency/contention at the start
        local = []
        for _ in range(per_producer):
            token_pos = next(counter)
            router.route(_record(point, token_pos))
            local.append(token_pos)
        with submitted_lock:
            submitted.extend(local)

    threads = [threading.Thread(target=_produce) for _ in range(num_producers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return submitted


def test_drop_oldest_under_concurrent_producers_accounts_for_every_record(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 0.002})  # deliberately slow vs. producer throughput
    router = Router(registry, default_overflow_policy=OverflowPolicy.DROP_OLDEST)
    point = make_extraction_point(name="ep-1", probe_type="slow", execution_mode=ExecutionMode.ASYNC, queue_depth=16)
    router.register_request("req-1", [point], make_request_ctx())

    submitted = _run_concurrent_producers(router, point, num_producers=8, per_producer=150)

    # _run_concurrent_producers has already joined every producer thread,
    # so no further put() can happen -- drop_count is final here, not
    # just a floor, even though it's read before end_request's drain.
    drop_count = router.get_metrics("req-1", "ep-1").drop_count

    results = router.end_request("req-1")
    router.shutdown()

    received = [r.token_pos for r in results["ep-1"].verdict]
    assert len(received) == len(set(received))  # no duplicates
    assert set(received) <= set(submitted)  # nothing received that wasn't submitted
    assert len(received) < len(submitted)  # a slow probe vs. 8 fast producers must genuinely overflow
    # No record is unaccounted for: every submitted item was either
    # received or evicted -- never both, never neither.
    assert len(received) + drop_count == len(submitted)


def test_drop_newest_under_concurrent_producers_accounts_for_every_record(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 0.002})
    router = Router(registry, default_overflow_policy=OverflowPolicy.DROP_NEWEST)
    point = make_extraction_point(name="ep-1", probe_type="slow", execution_mode=ExecutionMode.ASYNC, queue_depth=16)
    router.register_request("req-1", [point], make_request_ctx())

    submitted = _run_concurrent_producers(router, point, num_producers=8, per_producer=150)
    drop_count = router.get_metrics("req-1", "ep-1").drop_count  # final -- see sibling drop_oldest test's comment

    results = router.end_request("req-1")
    router.shutdown()

    received = [r.token_pos for r in results["ep-1"].verdict]
    assert len(received) == len(set(received))
    assert set(received) <= set(submitted)
    assert len(received) < len(submitted)  # a slow probe vs. 8 fast producers must genuinely overflow
    assert len(received) + drop_count == len(submitted)


def test_block_policy_zero_loss_under_concurrent_producers(probe_registry, make_request_ctx):
    """Under BLOCK, put() always eventually succeeds (never dropped) --
    confirm concurrent producers racing a slow consumer never lose or
    duplicate a single record."""
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 0.001})
    router = Router(registry, default_overflow_policy=OverflowPolicy.BLOCK)
    point = make_extraction_point(name="ep-1", probe_type="slow", execution_mode=ExecutionMode.ASYNC, queue_depth=4)
    router.register_request("req-1", [point], make_request_ctx())

    submitted = _run_concurrent_producers(router, point, num_producers=6, per_producer=60)

    results = router.end_request("req-1")
    router.shutdown()

    received = [r.token_pos for r in results["ep-1"].verdict]
    assert sorted(received) == sorted(submitted)


def test_block_policy_stalls_producer_thread_until_consumer_frees_space(probe_registry, make_request_ctx):
    """Backpressure: with execution_mode=async + overflow_policy=BLOCK,
    an adapter's generate() loop calling route() in a tight loop must
    actually stall once the queue is full, rather than the queue silently
    growing unbounded (which a drop_* policy would instead handle by
    discarding, and which BLOCK is specifically chosen to avoid).

    Uses GatedProbe rather than a timed sleep so "the queue is exactly
    full and the worker is busy" is a fact this test knows for certain
    (via `started`), not something it infers from a race-prone sleep.
    """
    registry = dict(probe_registry)
    registry["gated"] = ProbeFactory(GatedProbe, {})
    router = Router(registry, default_overflow_policy=OverflowPolicy.BLOCK)
    point = make_extraction_point(name="ep-1", probe_type="gated", execution_mode=ExecutionMode.ASYNC, queue_depth=2)
    router.register_request("req-1", [point], make_request_ctx())
    probe = router.get_probe("req-1", "ep-1")

    router.route(_record(point, token_pos=0))
    assert probe.started.wait(timeout=2)  # worker now blocked inside on_activation; queue is guaranteed empty

    # Fill the queue_depth=2 buffer -- both must return immediately, since
    # there's room and nothing yet to wait for.
    start = time.monotonic()
    router.route(_record(point, token_pos=1))
    router.route(_record(point, token_pos=2))
    assert time.monotonic() - start < 0.1

    # No free slot remains and the worker is still gated on item 0 -- the
    # next route() call must block rather than the queue silently growing
    # to size 3 (which a drop_* policy would instead handle by discarding
    # an item, not by blocking).
    unblocked = threading.Event()

    def _producer():
        router.route(_record(point, token_pos=3))
        unblocked.set()

    t = threading.Thread(target=_producer)
    t.start()
    time.sleep(0.2)
    assert not unblocked.is_set()  # still stalled: this is the backpressure the test exists to prove
    assert router.get_metrics("req-1", "ep-1").queue_depth <= 2  # never grew past the configured bound

    probe.release()
    t.join(timeout=2)
    assert unblocked.is_set()  # unblocked once the worker drained a slot

    results = router.end_request("req-1")
    router.shutdown()
    assert sorted(r.token_pos for r in results["ep-1"].verdict) == [0, 1, 2, 3]  # BLOCK: nothing was ever dropped

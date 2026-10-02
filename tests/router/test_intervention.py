import time

import pytest

from tests.router._helpers import FakeLogSink, FlakyProbe, ImmediateAbortProbe, SlowProbe, make_extraction_point
from undercurrent.core import ActivationRecord
from undercurrent.router import ProbeAction, ProbeFactory, Router, RouterError
from undercurrent.spec import ExecutionMode, InterventionMode, InterventionPolicy, ProbeKind, TimeoutAction


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


def test_block_until_signal_returns_real_signal_when_probe_is_fast(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["immediate_abort"] = ProbeFactory(ImmediateAbortProbe, {})
    router = Router(registry)

    point = make_extraction_point(
        name="ep-1",
        probe_type="immediate_abort",
        intervention=InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=500),
    )
    router.register_request("req-1", [point], make_request_ctx())

    signal = router.route(_record_for(point))

    assert signal.action == ProbeAction.ABORT
    assert signal.metadata["reason"] == "immediate_abort_probe"
    router.end_request("req-1")


def test_block_until_signal_times_out_and_falls_back_to_continue(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 1.0})
    router = Router(registry)

    point = make_extraction_point(
        name="ep-1",
        probe_type="slow",
        intervention=InterventionPolicy(
            mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=50, on_timeout=TimeoutAction.CONTINUE
        ),
    )
    router.register_request("req-1", [point], make_request_ctx())

    start = time.monotonic()
    signal = router.route(_record_for(point))
    elapsed = time.monotonic() - start

    assert elapsed < 0.5  # bounded by timeout_ms, not the probe's 1s delay
    assert signal.action == ProbeAction.CONTINUE
    assert signal.metadata["intervention_fallback"] is True
    assert signal.metadata["reason"] == "timeout"

    router.end_request("req-1")
    router.shutdown()  # lets the still-running SlowProbe call finish harmlessly


def test_block_until_signal_times_out_and_falls_back_to_abort(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 1.0})
    router = Router(registry)

    point = make_extraction_point(
        name="ep-1",
        probe_type="slow",
        intervention=InterventionPolicy(
            mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=50, on_timeout=TimeoutAction.ABORT
        ),
    )
    router.register_request("req-1", [point], make_request_ctx())

    signal = router.route(_record_for(point))

    assert signal.action == ProbeAction.ABORT
    assert signal.metadata["reason"] == "timeout"

    router.end_request("req-1")
    router.shutdown()


def test_block_until_signal_exception_falls_back(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["flaky"] = ProbeFactory(FlakyProbe, {"raise_on": lambda record: True})
    router = Router(registry)

    point = make_extraction_point(
        name="ep-1",
        probe_type="flaky",
        intervention=InterventionPolicy(
            mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=500, on_timeout=TimeoutAction.ABORT
        ),
    )
    router.register_request("req-1", [point], make_request_ctx())

    signal = router.route(_record_for(point))

    assert signal.action == ProbeAction.ABORT
    assert signal.metadata["reason"] == "exception"
    assert signal.metadata["error_type"] == "ValueError"

    router.end_request("req-1")


def test_router_level_default_intervention_applies_when_point_has_none(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 1.0})
    router = Router(
        registry,
        default_intervention_policy=InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=50),
    )

    point = make_extraction_point(name="ep-1", probe_type="slow")  # intervention=None -> inherits router default
    router.register_request("req-1", [point], make_request_ctx())

    start = time.monotonic()
    signal = router.route(_record_for(point))
    elapsed = time.monotonic() - start

    assert elapsed < 0.5
    assert signal.metadata["intervention_fallback"] is True

    router.end_request("req-1")
    router.shutdown()


def test_point_level_intervention_overrides_router_default(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["immediate_abort"] = ProbeFactory(ImmediateAbortProbe, {})
    router = Router(
        registry,
        default_intervention_policy=InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=50),
    )

    # Explicit reject at the point level overrides the router-wide block_until_signal default.
    point = make_extraction_point(
        name="ep-1", probe_type="immediate_abort", intervention=InterventionPolicy(mode=InterventionMode.REJECT)
    )
    router.register_request("req-1", [point], make_request_ctx())

    signal = router.route(_record_for(point))

    assert signal.action == ProbeAction.ABORT  # real signal, not a fallback
    assert "intervention_fallback" not in signal.metadata

    router.end_request("req-1")


def test_async_with_non_default_intervention_rejected_at_registration(probe_registry, make_request_ctx):
    # Requirement: async cannot participate in synchronous intervention --
    # any async extraction point whose *effective* InterventionPolicy isn't
    # the no-op default (mode=reject) must be rejected at register_request
    # time. undercurrent.spec's parser already rejects this combination at
    # parse time (see tests/spec/test_intervention.py); this proves
    # the router re-checks it defensively too, for a hand-built
    # ExtractionPoint that bypasses the parser entirely.
    point = make_extraction_point(
        probe_type="trajectory_score",
        probe_kind=ProbeKind.TRAJECTORY,
        execution_mode=ExecutionMode.ASYNC,
        queue_depth=4,
        intervention=InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=100),
    )
    router = Router(probe_registry)

    with pytest.raises(RouterError, match="execution_mode=async"):
        router.register_request("req-1", [point], make_request_ctx())


def test_async_with_non_default_router_level_default_rejected_at_registration(probe_registry, make_request_ctx):
    # Same rule, but the non-default policy comes from the router-level
    # default rather than the point itself -- the async point set no
    # intervention of its own, so it inherits the router's default, which
    # is block_until_signal here. It must still be rejected.
    point = make_extraction_point(
        probe_type="trajectory_score",
        probe_kind=ProbeKind.TRAJECTORY,
        execution_mode=ExecutionMode.ASYNC,
        queue_depth=4,
    )
    router = Router(
        probe_registry,
        default_intervention_policy=InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=100),
    )

    with pytest.raises(RouterError, match="execution_mode=async"):
        router.register_request("req-1", [point], make_request_ctx())


def test_async_explicit_reject_intervention_is_allowed(probe_registry, make_request_ctx):
    point = make_extraction_point(
        probe_type="trajectory_score",
        probe_kind=ProbeKind.TRAJECTORY,
        execution_mode=ExecutionMode.ASYNC,
        queue_depth=4,
        intervention=InterventionPolicy(mode=InterventionMode.REJECT),
    )
    router = Router(probe_registry)

    router.register_request("req-1", [point], make_request_ctx())  # must not raise
    router.end_request("req-1")
    router.shutdown()


def test_circuit_breaker_trips_after_n_consecutive_timeouts_and_downgrades_future_dispatch(
    probe_registry, make_request_ctx
):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 1.0})
    registry["immediate_abort"] = ProbeFactory(ImmediateAbortProbe, {})
    log_sink = FakeLogSink()
    router = Router(registry, circuit_breaker_threshold=3)
    router.attach_log_sink(log_sink)

    policy = InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=30)
    point = make_extraction_point(name="ep-1", probe_type="slow", intervention=policy)

    # Three separate requests, each timing out once -- "consecutive ...
    # across different requests" per the requirement.
    for i in range(3):
        request_id = f"req-{i}"
        router.register_request(request_id, [point], make_request_ctx(request_id=request_id))
        signal = router.route(_record_for(point, request_id=request_id))
        assert signal.metadata["intervention_fallback"] is True
        router.end_request(request_id)

    # Breaker should have tripped: a loud FLAG signal was forwarded to the log sink...
    flag_signals = [s for (_rid, _ep, s) in log_sink.signals if s.action == ProbeAction.FLAG]
    assert len(flag_signals) == 1
    assert flag_signals[0].metadata["circuit_breaker_tripped"] is True
    assert flag_signals[0].metadata["consecutive_failures"] == 3

    # ...and a NEW request for the same extraction point name is now
    # dispatched as plain reject, not block_until_signal: a fast probe's
    # real signal comes back immediately, with no fallback wrapper, even
    # though the point still asks for block_until_signal.
    downgraded_point = make_extraction_point(name="ep-1", probe_type="immediate_abort", intervention=policy)
    router.register_request("req-downgraded", [downgraded_point], make_request_ctx(request_id="req-downgraded"))
    signal = router.route(_record_for(downgraded_point, request_id="req-downgraded"))

    assert signal.action == ProbeAction.ABORT
    assert "intervention_fallback" not in signal.metadata  # real signal, not a timeout fallback

    router.end_request("req-downgraded")
    router.shutdown()


def test_circuit_breaker_resets_on_success_between_failures(probe_registry, make_request_ctx):
    registry = dict(probe_registry)
    registry["slow"] = ProbeFactory(SlowProbe, {"delay": 1.0})
    registry["immediate_abort"] = ProbeFactory(ImmediateAbortProbe, {})
    router = Router(registry, circuit_breaker_threshold=2)

    slow_policy = InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=30)
    fast_policy = InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=500)

    # One timeout, then one success, then one more timeout -- never two
    # CONSECUTIVE failures, so with threshold=2 the breaker should not trip.
    slow_point = make_extraction_point(name="ep-1", probe_type="slow", intervention=slow_policy)
    fast_point = make_extraction_point(name="ep-1", probe_type="immediate_abort", intervention=fast_policy)

    router.register_request("req-0", [slow_point], make_request_ctx(request_id="req-0"))
    router.route(_record_for(slow_point, request_id="req-0"))
    router.end_request("req-0")

    router.register_request("req-1", [fast_point], make_request_ctx(request_id="req-1"))
    router.route(_record_for(fast_point, request_id="req-1"))
    router.end_request("req-1")

    router.register_request("req-2", [slow_point], make_request_ctx(request_id="req-2"))
    signal = router.route(_record_for(slow_point, request_id="req-2"))
    router.end_request("req-2")

    # Still block_until_signal (not downgraded): the second timeout still
    # produced a fallback signal rather than the probe's own instant one.
    assert signal.metadata["intervention_fallback"] is True

    router.shutdown()

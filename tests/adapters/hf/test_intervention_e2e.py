"""End-to-end intervention (sync gate) test using the HF adapter -- the
adapter with the cleanest abort semantics (a single sequential decode loop;
see stopping_criteria.py's module docstring). Covers:

1. A TrajectorySafetyProbe-style probe (from content_safety_demo),
   registered in block_until_signal mode with a low intervention
   threshold: generation must actually halt within the expected token
   window, and the ProbeSignal metadata reporting the abort reason must be
   preserved and retrievable from end_request().
2. A timeout-forced fallback path: a deliberately slow probe with a short
   timeout_ms, confirming on_timeout fires (not the probe's own decision --
   the slow probe would say "continue" if given enough time, but never
   gets the chance).
"""

import time

from content_safety_demo import TrajectorySafetyProbe

from tests.adapters.hf._helpers import SlowThenContinueProbe, spy_end_request
from undercurrent.router import ProbeAction, ProbeFactory, Router
from undercurrent.spec import (
    ExecutionMode,
    ExtractionPoint,
    InterventionMode,
    InterventionPolicy,
    ProbeKind,
    TensorType,
    TimeoutAction,
    parse_position,
)


def _make_point(name, probe_type, intervention, layer=0):
    return ExtractionPoint(
        name=name,
        layers=(layer,),
        tensor_type=TensorType.RESIDUAL_STREAM,
        position=parse_position("generated[*]"),
        stride=None,
        until=None,
        probe_type=probe_type,
        probe_kind=ProbeKind.TRAJECTORY,
        execution_mode=ExecutionMode.INLINE,
        queue_depth=None,
        intervention=intervention,
    )


def test_block_until_signal_abort_halts_generation_and_preserves_signal_metadata(adapter, prompt):
    max_new_tokens = 20
    point = _make_point(
        name="safety",
        probe_type="trajectory_safety",
        # threshold=0.0: TrajectoryRecurrence's score is a sigmoid output,
        # always strictly > 0, so this guarantees an abort on the very
        # first generated activation -- deterministic, no need to tune a
        # real threshold against a random-initialized model's actual score
        # distribution.
        intervention=InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=2000),
    )
    router = Router(
        {"trajectory_safety": ProbeFactory(TrajectorySafetyProbe, {"layer": 0, "seed": 7, "threshold": 0.0})}
    )
    results = spy_end_request(router)

    request_id = "req-abort"
    adapter.register_extraction(request_id, [point])
    try:
        # min_new_tokens: the random-initialized model may otherwise emit EOS as
        # its very first token (it does with torch 2.1's RNG), and generation
        # would end before any generated-token activation reaches the probe.
        text = adapter.generate(
            request_id,
            prompt,
            {"max_new_tokens": max_new_tokens, "min_new_tokens": 2, "do_sample": False},
            router,
        )
    finally:
        router.shutdown()

    generated_token_count = len(text.split())
    # Generation must have halted well short of max_new_tokens -- "within
    # the expected token window" for an abort that fires on the very first
    # activation.
    assert generated_token_count < max_new_tokens

    result = results["results"]["safety"]
    assert result.verdict["aborted"] is True
    assert len(result.signal_history) >= 1
    abort_signal = result.signal_history[-1]
    assert abort_signal.action == ProbeAction.ABORT
    assert abort_signal.metadata["reason"] == "content_safety_threshold_exceeded"


def test_block_until_signal_timeout_falls_back_to_configured_action(adapter, prompt):
    max_new_tokens = 5
    point = _make_point(
        name="slow-check",
        probe_type="slow_then_continue",
        intervention=InterventionPolicy(
            mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=50, on_timeout=TimeoutAction.ABORT
        ),
    )
    router = Router({"slow_then_continue": ProbeFactory(SlowThenContinueProbe, {})})

    request_id = "req-timeout"
    adapter.register_extraction(request_id, [point])

    start = time.monotonic()
    try:
        text = adapter.generate(request_id, prompt, {"max_new_tokens": max_new_tokens, "do_sample": False}, router)
    finally:
        router.shutdown(wait=False)
    elapsed = time.monotonic() - start

    # Bounded by timeout_ms (~50ms) plus a token or two of generation
    # overhead -- nowhere near the slow probe's multi-second delay, proving
    # route() actually enforced timeout_ms rather than waiting the probe out.
    assert elapsed < 1.0

    generated_token_count = len(text.split())
    # on_timeout=abort must have actually stopped generation early: the
    # slow probe itself never gets a chance to answer within max_new_tokens
    # generations, so without the timeout fallback this would run to
    # max_new_tokens instead.
    assert generated_token_count < max_new_tokens

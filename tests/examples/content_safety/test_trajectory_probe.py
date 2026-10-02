import pytest
from content_safety_demo import TrajectorySafetyProbe

from undercurrent.core import ProbeAction, ProbeResult

LAYER = 8
SEED = 7


def _synthetic_unsafe_sequence(steps: int = 8, dim: int = 6):
    """A synthetic 'increasingly unsafe' activation stream: each step's
    vector is the same fixed direction scaled by a growing magnitude, so a
    downstream scorer sees a steadily ramping signal (rather than genuinely
    trained 'unsafe content' semantics, which this structural, random-init
    model has no notion of)."""
    base = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0][:dim]
    return [[v * (step + 1) for v in base] for step in range(steps)]


def test_running_score_accumulates_across_calls(make_record, make_request_ctx):
    activations = _synthetic_unsafe_sequence()
    probe = TrajectorySafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=2.0, seed=SEED)
    probe.on_start(make_request_ctx())

    scores = []
    for i, tensor in enumerate(activations):
        record = make_record(request_id="req-1", layer=LAYER, token_pos=i, tensor=tensor, is_generated=True)
        signal = probe.on_activation(record)
        assert signal is None  # threshold=2.0 is unreachable: scores are sigmoid-bounded to (0, 1)
        scores.append(probe.running_score)

    assert len(scores) == len(activations)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Re-running the identical (seed, input) pair must reproduce the exact
    # same trajectory -- the state genuinely accumulates deterministically
    # across calls rather than e.g. depending on call count alone.
    replay = TrajectorySafetyProbe.spawn("req-2", "ep-1", layer=LAYER, threshold=2.0, seed=SEED)
    replay.on_start(make_request_ctx(request_id="req-2"))
    replay_scores = []
    for i, tensor in enumerate(activations):
        record = make_record(request_id="req-2", layer=LAYER, token_pos=i, tensor=tensor, is_generated=True)
        replay.on_activation(record)
        replay_scores.append(replay.running_score)
    assert replay_scores == scores


def test_emits_abort_at_threshold_crossing_and_only_once(make_record, make_request_ctx):
    activations = _synthetic_unsafe_sequence()

    # Calibration pass: unreachable threshold, just record the full score
    # trajectory for this seed + input sequence.
    calibration = TrajectorySafetyProbe.spawn("req-cal", "ep-1", layer=LAYER, threshold=2.0, seed=SEED)
    calibration.on_start(make_request_ctx(request_id="req-cal"))
    scores = []
    for i, tensor in enumerate(activations):
        record = make_record(request_id="req-cal", layer=LAYER, token_pos=i, tensor=tensor, is_generated=True)
        calibration.on_activation(record)
        scores.append(calibration.running_score)

    # A threshold just under the peak score guarantees a crossing at the
    # first index the peak is reached, regardless of the actual (random,
    # seed-dependent) numeric values.
    threshold = max(scores) - 1e-6
    expected_crossing_index = min(i for i, s in enumerate(scores) if s > threshold)

    probe = TrajectorySafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=threshold, seed=SEED)
    probe.on_start(make_request_ctx())

    abort_index = None
    abort_signal = None
    for i, tensor in enumerate(activations):
        record = make_record(request_id="req-1", layer=LAYER, token_pos=i, tensor=tensor, is_generated=True)
        signal = probe.on_activation(record)
        if signal is not None:
            abort_index = i
            abort_signal = signal
            break

    assert abort_index == expected_crossing_index
    assert abort_signal.action == ProbeAction.ABORT
    assert abort_signal.metadata["reason"] == "content_safety_threshold_exceeded"
    assert abort_signal.metadata["score"] == pytest.approx(scores[expected_crossing_index])

    # Feed the rest of the sequence -- once aborted, no further signals.
    for tensor in activations[abort_index + 1 :]:
        record = make_record(request_id="req-1", layer=LAYER, tensor=tensor, is_generated=True)
        assert probe.on_activation(record) is None

    result = probe.on_end(make_request_ctx())
    assert isinstance(result, ProbeResult)
    assert result.verdict["aborted"] is True
    assert len(result.signal_history) == 1


def test_no_abort_while_below_threshold(make_record, make_request_ctx):
    probe = TrajectorySafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=2.0, seed=SEED)
    probe.on_start(make_request_ctx())
    signal = probe.on_activation(make_record(layer=LAYER, tensor=[1.0, -1.0], is_generated=True))
    assert signal is None


def test_layer_mismatch_raises(make_record, make_request_ctx):
    probe = TrajectorySafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=0.5, seed=SEED)
    probe.on_start(make_request_ctx())
    with pytest.raises(ValueError, match="layer"):
        probe.on_activation(make_record(layer=LAYER + 1, is_generated=True))


def test_on_end_well_formed_with_zero_activations(make_request_ctx):
    probe = TrajectorySafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=0.5)
    probe.on_start(make_request_ctx())
    result = probe.on_end(make_request_ctx())

    assert isinstance(result, ProbeResult)
    assert result.request_id == "req-1"
    assert result.extraction_point_name == "ep-1"
    assert result.verdict == {"final_score": 0.0, "count": 0, "aborted": False, "score_history": []}
    assert result.signal_history == []

import pytest
from content_safety_demo import SingleTokenSafetyProbe

from undercurrent.core import ProbeAction, ProbeResult

LAYER = 12


def test_flags_above_threshold_and_continues_below(make_record, make_request_ctx):
    # Same seed -> identical random weight init -> identical score for the
    # same input, so bracketing a fixed threshold around the observed score
    # deterministically exercises both the flagged and unflagged branches.
    probe = SingleTokenSafetyProbe.spawn("req-obs", "ep-1", layer=LAYER, threshold=0.5, seed=42)
    probe.on_start(make_request_ctx(request_id="req-obs"))
    record = make_record(request_id="req-obs", layer=LAYER, tensor=[3.0, -1.0, 2.0, 0.5, -2.0], is_generated=False)
    observed_signal = probe.on_activation(record)
    observed_score = observed_signal.confidence

    above = SingleTokenSafetyProbe.spawn("req-above", "ep-1", layer=LAYER, threshold=observed_score - 1e-6, seed=42)
    above.on_start(make_request_ctx(request_id="req-above"))
    signal_above = above.on_activation(
        make_record(request_id="req-above", layer=LAYER, tensor=[3.0, -1.0, 2.0, 0.5, -2.0])
    )
    assert signal_above.action == ProbeAction.FLAG
    assert signal_above.metadata["flagged"] is True
    result_above = above.on_end(make_request_ctx(request_id="req-above"))
    assert result_above.verdict["flagged"] is True

    below = SingleTokenSafetyProbe.spawn("req-below", "ep-1", layer=LAYER, threshold=observed_score + 1e-6, seed=42)
    below.on_start(make_request_ctx(request_id="req-below"))
    signal_below = below.on_activation(
        make_record(request_id="req-below", layer=LAYER, tensor=[3.0, -1.0, 2.0, 0.5, -2.0])
    )
    assert signal_below.action == ProbeAction.CONTINUE
    assert signal_below.metadata["flagged"] is False
    result_below = below.on_end(make_request_ctx(request_id="req-below"))
    assert result_below.verdict["flagged"] is False


def test_score_in_unit_interval(make_record, make_request_ctx):
    probe = SingleTokenSafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=0.5, seed=1)
    probe.on_start(make_request_ctx())
    signal = probe.on_activation(make_record(layer=LAYER, tensor=[100.0, -100.0, 50.0]))
    assert 0.0 <= signal.confidence <= 1.0


def test_second_on_activation_call_raises(make_record, make_request_ctx):
    probe = SingleTokenSafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=0.5, seed=1)
    probe.on_start(make_request_ctx())
    probe.on_activation(make_record(layer=LAYER))
    with pytest.raises(RuntimeError, match="second on_activation"):
        probe.on_activation(make_record(layer=LAYER, token_pos=1))


def test_layer_mismatch_raises(make_record, make_request_ctx):
    probe = SingleTokenSafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=0.5, seed=1)
    probe.on_start(make_request_ctx())
    with pytest.raises(ValueError, match="layer"):
        probe.on_activation(make_record(layer=LAYER + 1))


def test_on_end_well_formed_when_never_activated(make_request_ctx):
    probe = SingleTokenSafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=0.5)
    probe.on_start(make_request_ctx())
    result = probe.on_end(make_request_ctx())

    assert isinstance(result, ProbeResult)
    assert result.request_id == "req-1"
    assert result.extraction_point_name == "ep-1"
    assert result.verdict == {"score": None, "flagged": False, "layer": LAYER, "token_pos": None}
    assert result.signal_history == []


def test_on_end_matches_on_activation_verdict(make_record, make_request_ctx):
    probe = SingleTokenSafetyProbe.spawn("req-1", "ep-1", layer=LAYER, threshold=0.5, seed=7)
    probe.on_start(make_request_ctx())
    signal = probe.on_activation(make_record(layer=LAYER, token_pos=3, tensor=[1.0, 2.0, 3.0]))

    result = probe.on_end(make_request_ctx())
    assert result.verdict["score"] == pytest.approx(signal.confidence)
    assert result.verdict["flagged"] == signal.metadata["flagged"]
    assert result.verdict["token_pos"] == 3
    assert len(result.signal_history) == 1

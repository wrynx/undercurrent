import pytest

from undercurrent.core import ProbeResult, RequestContext
from undercurrent.core.examples import MLPClassifierProbe, TrajectoryScoreProbe


@pytest.mark.parametrize("probe_cls", [MLPClassifierProbe, TrajectoryScoreProbe])
def test_on_end_returns_result_with_zero_activations(probe_cls):
    probe = probe_cls.spawn("req-1", "ep-1")
    request_ctx = RequestContext(request_id="req-1", prompt_metadata={}, extraction_point_config=None)

    probe.on_start(request_ctx)
    result = probe.on_end(request_ctx)

    assert isinstance(result, ProbeResult)
    assert result.request_id == "req-1"
    assert result.extraction_point_name == "ep-1"
    assert result.signal_history == []


def test_mlp_classifier_verdict_none_before_any_activation():
    probe = MLPClassifierProbe.spawn("req-1", "ep-1")
    request_ctx = RequestContext(request_id="req-1", prompt_metadata={}, extraction_point_config=None)
    result = probe.on_end(request_ctx)
    assert result.verdict is None


def test_mlp_classifier_on_activation_returns_signal_and_verdict(make_record):
    probe = MLPClassifierProbe.spawn("req-1", "ep-1", num_classes=3)
    signal = probe.on_activation(make_record(tensor=[1.0, 2.0, 3.0]))
    assert signal is not None
    assert signal.metadata["predicted_class"] in (0, 1, 2)

    request_ctx = RequestContext(request_id="req-1", prompt_metadata={}, extraction_point_config=None)
    result = probe.on_end(request_ctx)
    assert result.verdict["predicted_class"] == signal.metadata["predicted_class"]
    assert len(result.verdict["logits"]) == 3
    assert len(result.signal_history) == 1

from undercurrent.core import ProbeAction, ProbeResult, RequestContext
from undercurrent.core.examples import TrajectoryScoreProbe


def test_running_mean_accumulates_across_calls(make_record):
    probe = TrajectoryScoreProbe.spawn("req-1", "ep-1", threshold=1.0)

    probe.on_activation(make_record(tensor=[1.0, 1.0], is_generated=True))  # score 1.0
    assert probe.running_mean == 1.0
    assert probe._count == 1

    probe.on_activation(make_record(tensor=[3.0, 3.0], is_generated=True))  # score 3.0
    # running mean of [1.0, 3.0] = 2.0
    assert probe.running_mean == 2.0
    assert probe._count == 2

    probe.on_activation(make_record(tensor=[2.0, 2.0], is_generated=True))  # score 2.0
    # running mean of [1.0, 3.0, 2.0] = 2.0
    assert probe.running_mean == 2.0
    assert probe._count == 3


def test_no_signal_while_below_threshold(make_record):
    probe = TrajectoryScoreProbe.spawn("req-1", "ep-1", threshold=100.0)
    signal = probe.on_activation(make_record(tensor=[1.0], is_generated=True))
    assert signal is None
    assert probe._signal_history == []


def test_abort_signal_emitted_once_threshold_crossed(make_record):
    probe = TrajectoryScoreProbe.spawn("req-1", "ep-1", threshold=2.0)

    signal_1 = probe.on_activation(make_record(tensor=[1.0], is_generated=True))  # mean 1.0
    assert signal_1 is None

    signal_2 = probe.on_activation(make_record(tensor=[5.0], is_generated=True))  # mean 3.0 >= 2.0
    assert signal_2 is not None
    assert signal_2.action == ProbeAction.ABORT
    assert signal_2.confidence == 3.0

    # Once aborted, subsequent activations must not re-emit abort signals.
    signal_3 = probe.on_activation(make_record(tensor=[5.0], is_generated=True))
    assert signal_3 is None

    assert len(probe._signal_history) == 1


def test_verdict_only_finalized_in_on_end(make_record):
    probe = TrajectoryScoreProbe.spawn("req-1", "ep-1", threshold=100.0)
    probe.on_activation(make_record(tensor=[4.0], is_generated=True))
    probe.on_activation(make_record(tensor=[6.0], is_generated=True))

    request_ctx = RequestContext(request_id="req-1", prompt_metadata={}, extraction_point_config=None)
    result = probe.on_end(request_ctx)

    assert isinstance(result, ProbeResult)
    assert result.verdict == {"final_mean": 5.0, "count": 2, "aborted": False}
    assert result.request_id == "req-1"
    assert result.extraction_point_name == "ep-1"

import pytest
from content_safety_demo import SingleTokenSafetyProbe, TrajectorySafetyProbe

from undercurrent.core import Probe, ProbeFactory


@pytest.mark.parametrize("probe_cls", [SingleTokenSafetyProbe, TrajectorySafetyProbe])
def test_spawn_returns_distinct_instances(probe_cls):
    a = probe_cls.spawn("req-1", "ep-1", layer=5)
    b = probe_cls.spawn("req-2", "ep-1", layer=5)
    assert a is not b
    assert a.request_id == "req-1"
    assert b.request_id == "req-2"


def test_spawned_trajectory_probes_do_not_share_running_state(make_record, make_request_ctx):
    a = TrajectorySafetyProbe.spawn("req-1", "ep-1", layer=5, threshold=2.0, seed=1)
    b = TrajectorySafetyProbe.spawn("req-2", "ep-1", layer=5, threshold=2.0, seed=1)
    a.on_start(make_request_ctx(request_id="req-1"))
    b.on_start(make_request_ctx(request_id="req-2"))

    a.on_activation(make_record(request_id="req-1", layer=5, tensor=[3.0, 3.0], is_generated=True))

    assert a._count == 1
    assert b._count == 0  # unaffected by a's activation
    assert a._score_history is not b._score_history
    assert a._signal_history is not b._signal_history


def test_probe_kwargs_are_not_shared_between_spawns():
    a = SingleTokenSafetyProbe.spawn("req-1", "ep-1", layer=5, threshold=0.3)
    b = SingleTokenSafetyProbe.spawn("req-2", "ep-1", layer=9, threshold=0.9)
    assert a._layer == 5 and a._threshold == 0.3
    assert b._layer == 9 and b._threshold == 0.9


@pytest.mark.parametrize("probe_cls", [SingleTokenSafetyProbe, TrajectorySafetyProbe])
def test_probe_factory_spawns_with_bound_kwargs(probe_cls):
    factory = ProbeFactory(probe_cls=probe_cls, probe_kwargs={"layer": 12, "threshold": 0.7})
    probe = factory.spawn("req-1", "ep-1")
    assert isinstance(probe, probe_cls)
    assert probe._layer == 12
    assert probe._threshold == 0.7
    assert probe.request_id == "req-1"
    assert probe.extraction_point_name == "ep-1"


@pytest.mark.parametrize("probe_cls", [SingleTokenSafetyProbe, TrajectorySafetyProbe])
def test_accessing_identity_before_spawn_raises(probe_cls):
    # Direct construction is allowed for isolated unit testing (as the other
    # test modules in this package do), but identity properties must not
    # silently return garbage before .spawn() has run.
    probe = probe_cls(layer=5)
    with pytest.raises(RuntimeError):
        probe.request_id
    with pytest.raises(RuntimeError):
        probe.extraction_point_name


def test_spawning_abstract_probe_rejected():
    with pytest.raises(TypeError):
        Probe.spawn("req-1", "ep-1")

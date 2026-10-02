import pytest

from undercurrent.core import Probe, ProbeFactory, ProbeResult
from undercurrent.core.examples import MLPClassifierProbe, TrajectoryScoreProbe


def test_spawn_returns_distinct_instances():
    a = TrajectoryScoreProbe.spawn("req-1", "ep-1")
    b = TrajectoryScoreProbe.spawn("req-2", "ep-1")
    assert a is not b


def test_spawned_instances_do_not_share_state(make_record):
    a = TrajectoryScoreProbe.spawn("req-1", "ep-1", threshold=0.5)
    b = TrajectoryScoreProbe.spawn("req-2", "ep-1", threshold=0.5)

    a.on_activation(make_record(request_id="req-1", tensor=[10.0, 10.0], is_generated=True))

    assert a.running_mean == 10.0
    assert b.running_mean == 0.0  # unaffected by a's activation
    assert a._signal_history is not b._signal_history


def test_spawn_sets_identity_per_instance():
    a = TrajectoryScoreProbe.spawn("req-1", "ep-A")
    b = TrajectoryScoreProbe.spawn("req-2", "ep-B")
    assert a.request_id == "req-1"
    assert a.extraction_point_name == "ep-A"
    assert b.request_id == "req-2"
    assert b.extraction_point_name == "ep-B"


def test_probe_kwargs_are_not_shared_between_spawns():
    a = MLPClassifierProbe.spawn("req-1", "ep-1", num_classes=3)
    b = MLPClassifierProbe.spawn("req-2", "ep-1", num_classes=5)
    assert a._num_classes == 3
    assert b._num_classes == 5


def test_spawning_abstract_probe_rejected():
    with pytest.raises(TypeError):
        Probe.spawn("req-1", "ep-1")


def test_spawn_requires_nonempty_ids():
    with pytest.raises(ValueError):
        TrajectoryScoreProbe.spawn("", "ep-1")
    with pytest.raises(ValueError):
        TrajectoryScoreProbe.spawn("req-1", "")


def test_accessing_identity_before_spawn_raises():
    # Direct construction is allowed (useful for unit-testing a single
    # method) but identity properties must not silently return garbage.
    probe = TrajectoryScoreProbe()
    with pytest.raises(RuntimeError):
        probe.request_id
    with pytest.raises(RuntimeError):
        probe.extraction_point_name


def test_probe_subclass_with_mutable_class_attribute_rejected():
    with pytest.raises(TypeError, match="class-level"):

        class BadProbe(Probe):
            probe_kind = "trajectory"
            _shared_cache = {}

            def on_start(self, request_ctx):
                pass

            def on_activation(self, record):
                return None

            def on_end(self, request_ctx):
                return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=None)


def test_probe_subclass_with_invalid_probe_kind_rejected():
    with pytest.raises(TypeError, match="probe_kind"):

        class BadKindProbe(Probe):
            probe_kind = "not_a_real_kind"

            def on_start(self, request_ctx):
                pass

            def on_activation(self, record):
                return None

            def on_end(self, request_ctx):
                return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=None)


def test_probe_factory_spawns_with_bound_kwargs():
    factory = ProbeFactory(probe_cls=MLPClassifierProbe, probe_kwargs={"num_classes": 4})
    probe = factory.spawn("req-1", "ep-1")
    assert isinstance(probe, MLPClassifierProbe)
    assert probe._num_classes == 4
    assert probe.request_id == "req-1"

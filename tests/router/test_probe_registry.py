"""Router <-> probe registry wiring: Router() with no registry, a class or a
factory in an explicit mapping, late registration, and probe_kind checks."""

import pytest

from tests.router._helpers import make_extraction_point
from undercurrent.core import ProbeFactory, ProbeRegistry, register_probe, unregister_probe
from undercurrent.core.examples import MLPClassifierProbe, TrajectoryScoreProbe
from undercurrent.router import Router, RouterError
from undercurrent.spec import ProbeKind


@pytest.fixture
def router_factory():
    routers = []

    def make(*args, **kwargs):
        router = Router(*args, **kwargs)
        routers.append(router)
        return router

    yield make
    for router in routers:
        router.shutdown()


@pytest.fixture
def default_registered():
    """Register into the default registry for one test, then clean up."""
    names = []

    def register(name, *args, **kwargs):
        result = register_probe(name, *args, **kwargs)
        names.append(name)
        return result

    yield register
    for name in names:
        unregister_probe(name)


def test_router_without_registry_uses_default_registry(router_factory, default_registered, make_request_ctx):
    default_registered("_test_router_default", TrajectoryScoreProbe, threshold=0.7)
    router = router_factory()

    router.register_request("req-1", [make_extraction_point(probe_type="_test_router_default")], make_request_ctx())
    probe = router.get_probe("req-1", "ep-1")
    assert isinstance(probe, TrajectoryScoreProbe)


def test_router_without_registry_sees_late_registration(router_factory, default_registered, make_request_ctx):
    router = router_factory()
    point = make_extraction_point(probe_type="_test_router_late")

    with pytest.raises(RouterError, match="no probe registered"):
        router.register_request("req-1", [point], make_request_ctx())

    default_registered("_test_router_late", TrajectoryScoreProbe)
    router.register_request("req-1", [point], make_request_ctx())
    assert isinstance(router.get_probe("req-1", "ep-1"), TrajectoryScoreProbe)


def test_router_accepts_a_probe_class_in_the_mapping(router_factory, make_request_ctx):
    router = router_factory({"trajectory_score": TrajectoryScoreProbe})
    router.register_request("req-1", [make_extraction_point()], make_request_ctx())
    assert isinstance(router.get_probe("req-1", "ep-1"), TrajectoryScoreProbe)


def test_router_accepts_a_probe_factory_in_the_mapping(router_factory, make_request_ctx):
    router = router_factory({"trajectory_score": ProbeFactory(TrajectoryScoreProbe, {"threshold": 0.8})})
    router.register_request("req-1", [make_extraction_point()], make_request_ctx())
    assert isinstance(router.get_probe("req-1", "ep-1"), TrajectoryScoreProbe)


def test_explicit_mapping_is_authoritative(router_factory, default_registered, make_request_ctx):
    default_registered("_test_router_only_default", TrajectoryScoreProbe)
    router = router_factory({"trajectory_score": TrajectoryScoreProbe})

    with pytest.raises(RouterError, match=r"no probe registered.*Known probe types: 'trajectory_score'"):
        router.register_request(
            "req-1", [make_extraction_point(probe_type="_test_router_only_default")], make_request_ctx()
        )


def test_explicit_mapping_is_copied(router_factory, make_request_ctx):
    mapping = {"trajectory_score": TrajectoryScoreProbe}
    router = router_factory(mapping)
    mapping.clear()
    router.register_request("req-1", [make_extraction_point()], make_request_ctx())


def test_explicit_mapping_rejects_non_probe_values():
    with pytest.raises(TypeError, match=r"expected a subclass of undercurrent.core.Probe"):
        Router({"bad": object})


def test_router_accepts_an_isolated_probe_registry_live(router_factory, make_request_ctx):
    registry = ProbeRegistry(load_entry_points=False)
    router = router_factory(registry)
    registry.register("trajectory_score", TrajectoryScoreProbe)  # after the router was built

    router.register_request("req-1", [make_extraction_point()], make_request_ctx())
    assert isinstance(router.get_probe("req-1", "ep-1"), TrajectoryScoreProbe)


def test_unknown_probe_type_message_suggests_close_matches(router_factory, make_request_ctx):
    router = router_factory({"trajectory_score": TrajectoryScoreProbe})
    with pytest.raises(RouterError, match=r"extraction point 'ep-1'.*Did you mean: 'trajectory_score'"):
        router.register_request("req-1", [make_extraction_point(probe_type="trajectory_scor")], make_request_ctx())


def test_probe_kind_mismatch_rejected_before_anything_spawns(router_factory, make_request_ctx):
    router = router_factory({"mlp_classifier": MLPClassifierProbe, "trajectory_score": TrajectoryScoreProbe})
    good = make_extraction_point(name="good", probe_type="trajectory_score")
    # MLPClassifierProbe is single_shot, but the point declares trajectory.
    bad = make_extraction_point(name="bad", probe_type="mlp_classifier", probe_kind=ProbeKind.TRAJECTORY)

    with pytest.raises(RouterError, match=r"probe_kind='single_shot'.*declares probe_kind='trajectory'"):
        router.register_request("req-1", [good, bad], make_request_ctx())

    with pytest.raises(RouterError):
        router.get_probe("req-1", "good")

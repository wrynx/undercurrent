import pytest

from tests.router._helpers import make_extraction_point
from undercurrent.router import Router, RouterError
from undercurrent.spec import ExecutionMode, InterventionMode, InterventionPolicy, ProbeKind


def test_register_request_spawns_and_starts_probes(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    point = make_extraction_point(name="ep-1", probe_type="trajectory_score")
    request_ctx = make_request_ctx()

    router.register_request("req-1", [point], request_ctx)

    probe = router.get_probe("req-1", "ep-1")
    assert probe.request_id == "req-1"
    assert probe.extraction_point_name == "ep-1"


def test_duplicate_request_id_rejected(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    point = make_extraction_point()
    router.register_request("req-1", [point], make_request_ctx())

    with pytest.raises(RouterError, match="already registered"):
        router.register_request("req-1", [point], make_request_ctx())


def test_unknown_probe_type_rejected(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    point = make_extraction_point(probe_type="does_not_exist")

    with pytest.raises(RouterError, match="no probe registered"):
        router.register_request("req-1", [point], make_request_ctx())

    # nothing should have been left behind
    with pytest.raises(RouterError):
        router.get_probe("req-1", point.name)


def test_single_shot_async_rejected_defensively(probe_registry, make_request_ctx):
    # undercurrent.spec's own parser already rejects this combination, but the
    # router must not trust that every ExtractionPoint it sees came from
    # the parser -- construct one directly, bypassing parsing entirely.
    point = make_extraction_point(
        probe_type="mlp_classifier",
        probe_kind=ProbeKind.SINGLE_SHOT,
        execution_mode=ExecutionMode.ASYNC,
        queue_depth=4,
    )
    router = Router(probe_registry)

    with pytest.raises(RouterError, match="execution_mode=async"):
        router.register_request("req-1", [point], make_request_ctx())


def test_async_with_non_default_intervention_policy_rejected(probe_registry, make_request_ctx):
    # See tests/router/test_intervention.py for the fuller suite
    # (including the router-level-default variant and the circuit breaker);
    # this is the specific, explicit proof requirement #2 asks for: async
    # cannot participate in synchronous intervention, structurally, at
    # register_request() time -- not just at undercurrent.spec parse time.
    point = make_extraction_point(
        probe_type="trajectory_score",
        probe_kind=ProbeKind.TRAJECTORY,
        execution_mode=ExecutionMode.ASYNC,
        queue_depth=4,
        intervention=InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=100),
    )
    router = Router(probe_registry)

    with pytest.raises(RouterError, match="execution_mode=async cannot use an intervention policy"):
        router.register_request("req-1", [point], make_request_ctx())

    # nothing should have been left behind
    with pytest.raises(RouterError):
        router.get_probe("req-1", point.name)


def test_bad_extraction_point_leaves_no_partial_registration(probe_registry, make_request_ctx):
    good_point = make_extraction_point(name="good", probe_type="trajectory_score")
    bad_point = make_extraction_point(
        name="bad",
        probe_type="mlp_classifier",
        probe_kind=ProbeKind.SINGLE_SHOT,
        execution_mode=ExecutionMode.ASYNC,
        queue_depth=4,
    )
    router = Router(probe_registry)

    with pytest.raises(RouterError):
        router.register_request("req-1", [good_point, bad_point], make_request_ctx())

    # "good" must not have been left dangling even though it validated fine
    with pytest.raises(RouterError):
        router.get_probe("req-1", "good")


def test_multiple_extraction_points_registered_independently(probe_registry, make_request_ctx):
    router = Router(probe_registry)
    points = [
        make_extraction_point(name="ep-a", probe_type="trajectory_score"),
        make_extraction_point(name="ep-b", probe_type="mlp_classifier", probe_kind=ProbeKind.SINGLE_SHOT),
    ]
    router.register_request("req-1", points, make_request_ctx())

    probe_a = router.get_probe("req-1", "ep-a")
    probe_b = router.get_probe("req-1", "ep-b")
    assert probe_a is not probe_b
    assert probe_a.extraction_point_name == "ep-a"
    assert probe_b.extraction_point_name == "ep-b"

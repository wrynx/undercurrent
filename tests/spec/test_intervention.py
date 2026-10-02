import pytest

from undercurrent.spec import (
    InterventionMode,
    InterventionPolicy,
    SpecValidationError,
    TimeoutAction,
    parse_dict,
    parse_yaml,
    probe_spec_to_dict,
    to_yaml,
)


def _spec_with_intervention(intervention=None, execution_mode="inline", probe_kind="single_shot", **extra):
    point = {
        "name": "ep-1",
        "layers": 5,
        "tensor_type": "residual_stream",
        "position": "generated[*]" if probe_kind == "trajectory" else "prompt[-1]",
        "probe_type": "some_probe",
        "probe_kind": probe_kind,
        "execution_mode": execution_mode,
        **extra,
    }
    if intervention is not None:
        point["intervention"] = intervention
    return {"version": "1", "extraction_points": [point]}


def test_no_intervention_specified_resolves_to_none():
    spec = parse_dict(_spec_with_intervention())
    assert spec.extraction_points[0].intervention is None


def test_explicit_reject_resolves():
    spec = parse_dict(_spec_with_intervention(intervention={"mode": "reject"}))
    point = spec.extraction_points[0]
    assert point.intervention == InterventionPolicy(mode=InterventionMode.REJECT)


def test_block_until_signal_requires_timeout_ms():
    with pytest.raises(SpecValidationError, match="timeout_ms is required"):
        parse_dict(_spec_with_intervention(intervention={"mode": "block_until_signal"}))


def test_block_until_signal_rejects_non_positive_timeout():
    with pytest.raises(SpecValidationError, match="timeout_ms must be >= 1"):
        parse_dict(_spec_with_intervention(intervention={"mode": "block_until_signal", "timeout_ms": 0}))


def test_reject_mode_forbids_timeout_ms():
    with pytest.raises(SpecValidationError, match="only valid when mode='block_until_signal'"):
        parse_dict(_spec_with_intervention(intervention={"mode": "reject", "timeout_ms": 100}))


def test_reject_mode_forbids_non_default_on_timeout():
    with pytest.raises(SpecValidationError, match="only meaningful when mode='block_until_signal'"):
        parse_dict(_spec_with_intervention(intervention={"mode": "reject", "on_timeout": "abort"}))


def test_block_until_signal_resolves_with_abort_fallback():
    spec = parse_dict(
        _spec_with_intervention(intervention={"mode": "block_until_signal", "timeout_ms": 250, "on_timeout": "abort"})
    )
    point = spec.extraction_points[0]
    assert point.intervention == InterventionPolicy(
        mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=250, on_timeout=TimeoutAction.ABORT
    )


def test_async_rejects_non_default_intervention():
    with pytest.raises(SpecValidationError, match="async cannot participate in synchronous intervention"):
        parse_dict(
            _spec_with_intervention(
                execution_mode="async",
                probe_kind="trajectory",
                intervention={"mode": "block_until_signal", "timeout_ms": 100},
            )
        )


def test_async_allows_explicit_reject_intervention():
    spec = parse_dict(
        _spec_with_intervention(
            execution_mode="async",
            probe_kind="trajectory",
            intervention={"mode": "reject"},
        )
    )
    assert spec.extraction_points[0].intervention == InterventionPolicy(mode=InterventionMode.REJECT)


def test_intervention_round_trips_through_dict_and_yaml():
    original = _spec_with_intervention(
        intervention={"mode": "block_until_signal", "timeout_ms": 50, "on_timeout": "abort"}
    )
    spec = parse_dict(original)

    respec_from_dict = parse_dict(probe_spec_to_dict(spec))
    assert respec_from_dict == spec

    respec_from_yaml = parse_yaml(to_yaml(spec))
    assert respec_from_yaml == spec


@pytest.mark.parametrize("timeout_ms", [None, 0])
def test_resolved_block_until_signal_requires_timeout(timeout_ms):
    # Built directly in Python (bypassing InterventionPolicySpec), a
    # block_until_signal policy without a positive timeout used to be accepted
    # and then crash Router.route() with a TypeError on `None / 1000.0`.
    with pytest.raises(ValueError, match="timeout_ms"):
        InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=timeout_ms)


def test_resolved_reject_policy_needs_no_timeout():
    assert InterventionPolicy(mode=InterventionMode.REJECT).timeout_ms is None

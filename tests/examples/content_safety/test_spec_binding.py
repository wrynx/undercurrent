from pathlib import Path

import pytest
from content_safety_demo import SingleTokenSafetyProbe, TrajectorySafetyProbe
from content_safety_demo.spec_binding import probe_kwargs_from_extraction_point, resolve_probe_cls

from undercurrent.core import ProbeFactory, ProbeNotFoundError
from undercurrent.spec import ExecutionMode, ProbeKind, TensorType, load_yaml_file, parse_position
from undercurrent.spec.resolved import ExtractionPoint

EXAMPLE_SPEC_PATH = Path(__file__).resolve().parents[3] / "examples" / "content_safety" / "content_safety_llama.yaml"


def test_example_spec_parses_and_round_trips_through_resolve_probe_cls():
    spec = load_yaml_file(EXAMPLE_SPEC_PATH)
    assert set(spec.names) == {"prompt_safety_check", "generation_safety_trajectory"}

    single_shot_point = spec.get("prompt_safety_check")
    assert resolve_probe_cls(single_shot_point) is SingleTokenSafetyProbe

    trajectory_point = spec.get("generation_safety_trajectory")
    assert resolve_probe_cls(trajectory_point) is TrajectorySafetyProbe


def test_resolve_probe_cls_rejects_unknown_probe_type():
    point = ExtractionPoint(
        name="ep-1",
        layers=(1,),
        tensor_type=TensorType.RESIDUAL_STREAM,
        position=parse_position("prompt[-1]"),
        stride=None,
        until=None,
        probe_type="not_a_registered_probe",
        probe_kind=ProbeKind.SINGLE_SHOT,
        execution_mode=ExecutionMode.INLINE,
        queue_depth=None,
    )
    with pytest.raises(ProbeNotFoundError, match="no probe registered"):
        resolve_probe_cls(point)


def test_resolve_probe_cls_rejects_probe_kind_mismatch():
    # probe_type names the trajectory probe, but the spec (incorrectly)
    # declares single_shot -- exactly the wiring bug this helper exists to
    # catch before a request ever reaches the mismatched probe.
    point = ExtractionPoint(
        name="ep-1",
        layers=(1,),
        tensor_type=TensorType.RESIDUAL_STREAM,
        position=parse_position("prompt[-1]"),
        stride=None,
        until=None,
        probe_type="content_safety_trajectory",
        probe_kind=ProbeKind.SINGLE_SHOT,
        execution_mode=ExecutionMode.INLINE,
        queue_depth=None,
    )
    with pytest.raises(ValueError, match="probe_kind"):
        resolve_probe_cls(point)


def test_probe_kwargs_from_extraction_point_derives_layer_and_accepts_overrides():
    spec = load_yaml_file(EXAMPLE_SPEC_PATH)
    point = spec.get("generation_safety_trajectory")

    kwargs = probe_kwargs_from_extraction_point(point, threshold=0.9, seed=3)
    assert kwargs == {"layer": 16, "threshold": 0.9, "seed": 3}

    probe = TrajectorySafetyProbe.spawn("req-1", point.name, **kwargs)
    assert probe._layer == 16
    assert probe._threshold == 0.9


def test_probe_kwargs_from_extraction_point_rejects_multi_layer():
    point = ExtractionPoint(
        name="ep-multi",
        layers=(1, 2, 3),
        tensor_type=TensorType.RESIDUAL_STREAM,
        position=parse_position("generated[*]"),
        stride=None,
        until=None,
        probe_type="content_safety_trajectory",
        probe_kind=ProbeKind.TRAJECTORY,
        execution_mode=ExecutionMode.INLINE,
        queue_depth=None,
    )
    with pytest.raises(ValueError, match="single-layer"):
        probe_kwargs_from_extraction_point(point)


def test_probe_factory_built_from_resolved_spec(make_record, make_request_ctx):
    spec = load_yaml_file(EXAMPLE_SPEC_PATH)
    point = spec.get("prompt_safety_check")

    probe_cls = resolve_probe_cls(point)
    kwargs = probe_kwargs_from_extraction_point(point, threshold=0.5, seed=11)
    factory = ProbeFactory(probe_cls, kwargs)

    probe = factory.spawn("req-1", point.name)
    probe.on_start(make_request_ctx())
    signal = probe.on_activation(make_record(layer=16, tensor=[1.0, 2.0, 3.0]))
    assert signal is not None


def test_demo_probes_are_registered_under_their_spec_names():
    from content_safety_demo import CustomMLPProbe

    from undercurrent.core import get_probe_factory

    assert get_probe_factory("content_safety_single_token").probe_cls is SingleTokenSafetyProbe
    assert get_probe_factory("content_safety_trajectory").probe_cls is TrajectorySafetyProbe
    assert get_probe_factory("custom_mlp_safety_check").probe_cls is CustomMLPProbe

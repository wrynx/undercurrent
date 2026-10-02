import pytest

from undercurrent.spec import (
    ExecutionMode,
    PositionKind,
    ProbeKind,
    SpecValidationError,
    TensorType,
    UntilKind,
    parse_dict,
    parse_yaml,
)


def _base_point(**overrides):
    point = {
        "name": "layer10_last_prompt_token",
        "layers": 10,
        "tensor_type": "residual_stream",
        "position": "prompt[-1]",
        "probe_type": "linear_probe",
        "probe_kind": "single_shot",
    }
    point.update(overrides)
    return point


def test_valid_minimal_spec():
    spec = parse_dict({"extraction_points": [_base_point()]})
    assert len(spec) == 1
    point = spec.get("layer10_last_prompt_token")
    assert point.layers == (10,)
    assert point.tensor_type == TensorType.RESIDUAL_STREAM
    assert point.execution_mode == ExecutionMode.INLINE
    assert point.position.kind == PositionKind.PROMPT_INDEX


def test_valid_multi_layer_spec():
    spec = parse_dict({"extraction_points": [_base_point(layers=[3, 7, 11])]})
    assert spec.get("layer10_last_prompt_token").layers == (3, 7, 11)


def test_valid_yaml_string():
    text = """
    extraction_points:
      - name: probe_a
        layers: 5
        tensor_type: mlp_out
        position: "generated[*]"
        probe_type: trajectory_probe
        probe_kind: trajectory
        execution_mode: async
        queue_depth: 4
    """
    spec = parse_yaml(text)
    point = spec.get("probe_a")
    assert point.execution_mode == ExecutionMode.ASYNC
    assert point.queue_depth == 4
    assert point.probe_kind == ProbeKind.TRAJECTORY


def test_every_n_alias_for_stride():
    spec = parse_dict({"extraction_points": [_base_point(position="generated[*]", every_n=4, probe_kind="trajectory")]})
    assert spec.get("layer10_last_prompt_token").stride == 4


def test_single_shot_async_rejected():
    with pytest.raises(SpecValidationError, match="execution_mode='async'"):
        parse_dict({"extraction_points": [_base_point(probe_kind="single_shot", execution_mode="async")]})


def test_single_shot_async_error_is_not_a_warning():
    # There is no "parse with warnings" path -- an invalid combination must
    # raise, full stop.
    with pytest.raises(SpecValidationError):
        parse_dict({"extraction_points": [_base_point(probe_kind="single_shot", execution_mode="async")]})


def test_queue_depth_without_async_rejected():
    with pytest.raises(SpecValidationError, match="queue_depth"):
        parse_dict({"extraction_points": [_base_point(queue_depth=2)]})


def test_stride_on_point_selector_rejected():
    with pytest.raises(SpecValidationError, match="stride"):
        parse_dict({"extraction_points": [_base_point(stride=2)]})


def test_until_on_point_selector_rejected():
    with pytest.raises(SpecValidationError, match="until"):
        parse_dict({"extraction_points": [_base_point(until="fixed_count")]})


def test_stride_valid_on_continuous_selector():
    spec = parse_dict(
        {
            "extraction_points": [
                _base_point(
                    position="generated[*]",
                    stride=3,
                    probe_kind="trajectory",
                    until="generation_end",
                )
            ]
        }
    )
    point = spec.get("layer10_last_prompt_token")
    assert point.stride == 3
    assert point.until == UntilKind.GENERATION_END


def test_duplicate_names_rejected():
    with pytest.raises(SpecValidationError, match="duplicate"):
        parse_dict({"extraction_points": [_base_point(), _base_point()]})


def test_invalid_tensor_enum_rejected():
    with pytest.raises(SpecValidationError):
        parse_dict({"extraction_points": [_base_point(tensor_type="not_a_real_tensor")]})


def test_invalid_position_syntax_rejected():
    with pytest.raises(SpecValidationError):
        parse_dict({"extraction_points": [_base_point(position="prompt[]")]})


def test_negative_layer_rejected():
    with pytest.raises(SpecValidationError):
        parse_dict({"extraction_points": [_base_point(layers=-1)]})


def test_empty_layer_list_rejected():
    with pytest.raises(SpecValidationError):
        parse_dict({"extraction_points": [_base_point(layers=[])]})


def test_unknown_field_rejected():
    with pytest.raises(SpecValidationError):
        parse_dict({"extraction_points": [_base_point(bogus_field=123)]})


def test_missing_required_field_rejected():
    point = _base_point()
    del point["probe_type"]
    with pytest.raises(SpecValidationError, match="probe_type"):
        parse_dict({"extraction_points": [point]})


def test_invalid_yaml_syntax_rejected():
    with pytest.raises(SpecValidationError):
        parse_yaml("extraction_points: [this: is not, valid: yaml: at all")


def test_top_level_not_a_mapping_rejected():
    with pytest.raises(SpecValidationError):
        parse_yaml("- just\n- a\n- list\n")


def test_parser_annotations_resolve():
    # Regression: `_resolve_intervention` was annotated with `Optional[...]`
    # without importing it. Under `from __future__ import annotations` that
    # only fails when something evaluates the hints (get_type_hints, pydantic,
    # doc tooling), so check every function in the module resolves.
    import inspect
    import typing

    from undercurrent.spec import parser

    for _name, fn in inspect.getmembers(parser, inspect.isfunction):
        if fn.__module__ == parser.__name__:
            typing.get_type_hints(fn)  # raises NameError on an unresolvable name

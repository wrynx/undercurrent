"""Canonical spec vocabulary (`layers`, `tensor_type`), the deprecated
`layer`/`tensor` aliases, and the `probe_args` field."""

import pickle
import warnings
from pathlib import Path

import pytest

from undercurrent.spec import (
    ExecutionMode,
    ExtractionPoint,
    ProbeKind,
    SpecValidationError,
    TensorType,
    extraction_point_to_dict,
    load_yaml_file,
    parse_dict,
    parse_position,
    parse_yaml,
    probe_spec_to_dict,
    to_yaml,
)

EXAMPLE_SPECS = sorted((Path(__file__).resolve().parents[2] / "examples").rglob("*.yaml"))


def _point(**overrides):
    point = {
        "name": "ep",
        "layers": 3,
        "tensor_type": "residual_stream",
        "position": "prompt[-1]",
        "probe_type": "linear_probe",
        "probe_kind": "single_shot",
    }
    point.update(overrides)
    return {key: value for key, value in point.items() if value is not None}


def _deprecations(caught):
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


# -- canonical keys ----------------------------------------------------------


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_canonical_keys_parse_without_warning():
    spec = parse_dict({"extraction_points": [_point(layers=[1, 2], tensor_type="mlp_out")]})
    point = spec.get("ep")
    assert point.layers == (1, 2)
    assert point.tensor_type == TensorType.MLP_OUT


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_canonical_int_layers_resolves_to_tuple():
    assert parse_dict({"extraction_points": [_point(layers=7)]}).get("ep").layers == (7,)


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_canonical_keys_in_yaml():
    spec = parse_yaml(
        """
        extraction_points:
          - name: ep
            layers: [0, 4]
            tensor_type: attn_out
            position: "generated[*]"
            probe_type: t
            probe_kind: trajectory
        """
    )
    assert spec.get("ep").layers == (0, 4)
    assert spec.get("ep").tensor_type == TensorType.ATTN_OUT


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_serializer_emits_canonical_keys_only():
    data = extraction_point_to_dict(parse_dict({"extraction_points": [_point()]}).get("ep"))
    assert data["layers"] == 3
    assert data["tensor_type"] == "residual_stream"
    assert "layer" not in data
    assert "tensor" not in data
    assert "probe_args" not in data


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_python_construction_uses_canonical_names():
    point = ExtractionPoint(
        name="ep",
        layers=(2,),
        tensor_type=TensorType.RESIDUAL_STREAM,
        position=parse_position("prompt[-1]"),
        stride=None,
        until=None,
        probe_type="linear_probe",
        probe_kind=ProbeKind.SINGLE_SHOT,
        execution_mode=ExecutionMode.INLINE,
        queue_depth=None,
        probe_args={"threshold": 0.5},
    )
    assert point == parse_dict({"extraction_points": [_point(layers=2, probe_args={"threshold": 0.5})]}).get("ep")


# -- deprecated aliases ------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new", "value", "attr", "expected"),
    [
        ("layer", "layers", 5, "layers", (5,)),
        ("layer", "layers", [1, 2], "layers", (1, 2)),
        ("tensor", "tensor_type", "kv", "tensor_type", TensorType.KV),
    ],
)
def test_alias_parses_and_warns_once(old, new, value, attr, expected):
    point = _point(**{new: None, old: value})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = parse_dict({"extraction_points": [point]})

    assert getattr(spec.get("ep"), attr) == expected
    deprecations = _deprecations(caught)
    assert len(deprecations) == 1
    message = str(deprecations[0].message)
    assert "'ep'" in message
    assert f"'{old}'" in message
    assert f"'{new}'" in message


def test_both_aliases_on_several_points_warn_once_per_parse_call():
    points = [_point(name=name, layers=None, tensor_type=None, layer=1, tensor="mlp_out") for name in ("a", "b")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_yaml(to_yaml(parse_dict({"extraction_points": points})))
        assert len(_deprecations(caught)) == 1  # to_yaml/parse_yaml of the result: canonical, no warning
        parse_dict({"extraction_points": points})

    deprecations = _deprecations(caught)
    assert len(deprecations) == 2  # one per parse call that used old keys
    message = str(deprecations[1].message)
    for name in ("'a'", "'b'"):
        assert name in message


def test_alias_warning_is_attributed_to_the_caller():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_dict({"extraction_points": [_point(layers=None, layer=1)]})
    # Python's default filters only show DeprecationWarnings attributed to
    # __main__, so the warning must point at user code, not at pydantic.
    assert _deprecations(caught)[0].filename == __file__


def test_alias_in_yaml_warns():
    with pytest.warns(DeprecationWarning, match="'layer', rename it to 'layers'"):
        spec = parse_yaml(
            """
            extraction_points:
              - name: ep
                layer: 5
                tensor_type: mlp_out
                position: "prompt[-1]"
                probe_type: p
                probe_kind: single_shot
            """
        )
    assert spec.get("ep").layers == (5,)


@pytest.mark.parametrize(("old", "new", "old_value"), [("layer", "layers", 4), ("tensor", "tensor_type", "kv")])
def test_old_and_new_key_together_is_an_error(old, new, old_value):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(SpecValidationError, match=f"both '{old}' \\(deprecated\\) and '{new}'"):
            parse_dict({"extraction_points": [_point(**{old: old_value})]})


# -- probe_args --------------------------------------------------------------

PROBE_ARGS = {"threshold": 0.25, "model_path": "/tmp/probe.safetensors", "classes": ["a", "b"], "opts": {"k": 1}}


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_probe_args_default_empty():
    point = parse_dict({"extraction_points": [_point()]}).get("ep")
    assert dict(point.probe_args) == {}
    assert not point.probe_args


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_probe_args_round_trip_yaml():
    spec = parse_dict({"extraction_points": [_point(probe_args=PROBE_ARGS)]})
    assert dict(spec.get("ep").probe_args) == PROBE_ARGS
    text = to_yaml(spec)
    assert "probe_args:" in text
    assert parse_yaml(text) == spec


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_probe_args_round_trip_extraction_point_to_dict():
    point = parse_dict({"extraction_points": [_point(probe_args=PROBE_ARGS)]}).get("ep")
    data = extraction_point_to_dict(point)
    assert data["probe_args"] == PROBE_ARGS
    assert type(data["probe_args"]) is dict
    assert parse_dict({"extraction_points": [data]}).get("ep") == point


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_probe_args_survive_vllm_worker_rpc_coercion():
    worker_extension = pytest.importorskip("undercurrent.adapters.vllm.worker_extension")
    point = parse_dict({"extraction_points": [_point(probe_args=PROBE_ARGS)]}).get("ep")
    (coerced,) = worker_extension._coerce_extraction_points([extraction_point_to_dict(point)])
    assert coerced == point


def test_probe_args_is_immutable_and_copied():
    source = {"threshold": 0.5}
    point = parse_dict({"extraction_points": [_point(probe_args=source)]}).get("ep")
    source["threshold"] = 0.9
    assert point.probe_args["threshold"] == 0.5
    with pytest.raises(TypeError):
        point.probe_args["threshold"] = 1.0  # type: ignore[index]


def test_extraction_point_stays_hashable_and_picklable():
    point = parse_dict({"extraction_points": [_point(probe_args=PROBE_ARGS)]}).get("ep")
    hash(point)
    assert pickle.loads(pickle.dumps(point)) == point


def test_probe_args_must_be_a_mapping():
    with pytest.raises(SpecValidationError, match="probe_args"):
        parse_dict({"extraction_points": [_point(probe_args=[1, 2])]})


# -- example specs -----------------------------------------------------------


def test_example_specs_found():
    assert EXAMPLE_SPECS


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("path", EXAMPLE_SPECS, ids=lambda p: p.name)
def test_example_spec_round_trips(path):
    spec = load_yaml_file(path)
    assert parse_yaml(to_yaml(spec)) == spec
    assert parse_dict(probe_spec_to_dict(spec)) == spec

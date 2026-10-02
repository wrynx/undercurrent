"""Tests for undercurrent.spec.json_schema and the committed schema file."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pydantic
import pytest
import yaml
from packaging.version import Version

from undercurrent.spec import PositionSyntaxError, SpecValidationError, json_schema, parse_dict, parse_position
from undercurrent.spec.json_schema import POSITION_PATTERN, SCHEMA_ID

jsonschema = pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = REPO_ROOT / "schema" / "probe-spec.schema.json"
EXAMPLE_YAMLS = sorted(p for ext in ("*.yaml", "*.yml") for p in (REPO_ROOT / "examples").rglob(ext))
HEADER = f"# yaml-language-server: $schema={SCHEMA_ID}"


@pytest.fixture(scope="module")
def validator():
    schema = json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


_DROP = object()


def _point(**overrides):
    point = {
        "name": "p",
        "layers": 3,
        "tensor_type": "residual_stream",
        "position": "prompt[-1]",
        "probe_type": "linear_probe",
        "probe_kind": "single_shot",
    }
    point.update(overrides)
    return {k: v for k, v in point.items() if v is not _DROP}


def _parser_accepts(spec: dict) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parse_dict(spec)
    except SpecValidationError:
        return False
    return True


# -- the committed file -------------------------------------------------------

# The committed file is generated with pydantic>=2.11. Older pydantic emits an
# equivalent schema with a different shape (e.g. allOf-wrapped $refs), so the
# byte-for-byte comparisons only hold from 2.11 (the dependency-floors CI job
# runs pydantic 2.0).
needs_current_pydantic = pytest.mark.skipif(
    Version(pydantic.VERSION) < Version("2.11"),
    reason="schema/probe-spec.schema.json is generated with pydantic>=2.11",
)


@needs_current_pydantic
def test_committed_schema_is_up_to_date():
    expected = json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"
    assert SCHEMA_FILE.exists(), "schema/probe-spec.schema.json is missing; run `python scripts/generate_schema.py`"
    assert SCHEMA_FILE.read_text(encoding="utf-8") == expected, (
        "schema/probe-spec.schema.json is stale; run `python scripts/generate_schema.py` and commit the result"
    )


def test_output_is_deterministic():
    assert json.dumps(json_schema()) == json.dumps(json_schema())


def test_top_level_metadata():
    schema = json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "https://raw.githubusercontent.com/wrynx/undercurrent/main/schema/probe-spec.schema.json"
    assert schema["title"]
    assert schema["description"]


def test_every_property_has_a_description():
    schema = json_schema()
    objects = [schema, schema["$defs"]["ExtractionPointSpec"], schema["$defs"]["InterventionPolicySpec"]]
    for obj in objects:
        for name, prop in obj["properties"].items():
            assert prop.get("description"), f"property {name!r} has no description"


def test_examples_for_key_fields():
    props = json_schema()["$defs"]["ExtractionPointSpec"]["properties"]
    for field in ("tensor_type", "position", "probe_kind"):
        assert props[field]["examples"], field


def test_import_does_not_pull_in_jsonschema():
    code = "import sys, undercurrent.spec; sys.exit(int('jsonschema' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


@needs_current_pydantic
def test_generate_script_writes_the_same_file(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import generate_schema
    finally:
        sys.path.pop(0)
    out = tmp_path / "probe-spec.schema.json"
    monkeypatch.setattr(generate_schema, "SCHEMA_PATH", out)
    generate_schema.main()
    assert out.read_text(encoding="utf-8") == SCHEMA_FILE.read_text(encoding="utf-8")


# -- example specs ------------------------------------------------------------


def test_examples_exist():
    assert EXAMPLE_YAMLS, "no example YAMLs found under examples/"


@pytest.mark.parametrize("path", EXAMPLE_YAMLS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_example_yaml_validates(path, validator):
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == HEADER, f"{path} must start with the yaml-language-server schema header"
    data = yaml.safe_load(text)
    errors = [e.message for e in validator.iter_errors(data)]
    assert not errors, errors
    assert _parser_accepts(data)


# -- deprecated aliases -------------------------------------------------------


@pytest.mark.parametrize(
    "point",
    [
        _point(layers=_DROP, layer=3),
        _point(tensor_type=_DROP, tensor="mlp_out"),
        _point(position="generated[*]", probe_kind="trajectory", every_n=2),
        _point(layers=_DROP, layer=[1, 2], tensor_type=_DROP, tensor="kv"),
    ],
)
def test_deprecated_aliases_still_validate(point, validator):
    spec = {"extraction_points": [point]}
    assert list(validator.iter_errors(spec)) == []
    assert _parser_accepts(spec)


def test_deprecated_aliases_are_not_suggested():
    props = json_schema()["$defs"]["ExtractionPointSpec"]["properties"]
    for old in ("layer", "tensor", "every_n"):
        assert props[old]["deprecated"] is True
        assert props[old]["doNotSuggest"] is True
        assert "examples" not in props[old]
    for canonical in ("layers", "tensor_type", "stride"):
        assert "deprecated" not in props[canonical]


@pytest.mark.parametrize(
    "point",
    [
        _point(layer=3),  # both layers and layer
        _point(tensor="kv"),  # both tensor_type and tensor
        _point(position="generated[*]", probe_kind="trajectory", stride=2, every_n=2),
        _point(layers=_DROP),  # neither
    ],
)
def test_alias_conflicts_are_rejected(point, validator):
    spec = {"extraction_points": [point]}
    assert not validator.is_valid(spec)
    assert not _parser_accepts(spec)


# -- invalid specs ------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param({"extraction_points": [_point(colour="red")]}, id="unknown-point-key"),
        pytest.param({"extraction_points": [_point()], "extra": 1}, id="unknown-top-level-key"),
        pytest.param({"extraction_points": [_point(tensor_type="residual")]}, id="bad-tensor-type"),
        pytest.param({"extraction_points": [_point(probe_kind="one_shot")]}, id="bad-probe-kind"),
        pytest.param({"extraction_points": [_point(execution_mode="sync")]}, id="bad-execution-mode"),
        pytest.param({"extraction_points": [_point(layers=-1)]}, id="negative-layer"),
        pytest.param({"extraction_points": [_point(layers=[2, -1])]}, id="negative-layer-in-list"),
        pytest.param({"extraction_points": [_point(layers=[])]}, id="empty-layers"),
        pytest.param({"extraction_points": [_point(name=" ")]}, id="blank-name"),
        pytest.param({"extraction_points": [_point(position="promt[-1]")]}, id="position-typo"),
        pytest.param({"extraction_points": [_point(position="generated[1.5]")]}, id="position-float-index"),
        pytest.param({"extraction_points": [_point(stride=2)]}, id="stride-on-point-selector"),
        pytest.param({"extraction_points": [_point(position="generated[*]", stride=0)]}, id="stride-zero"),
        pytest.param({"extraction_points": [_point(until="generation_end")]}, id="until-on-point-selector"),
        pytest.param({"extraction_points": [_point(execution_mode="async")]}, id="async-single-shot"),
        pytest.param({"extraction_points": [_point(queue_depth=4)]}, id="queue-depth-inline"),
        pytest.param(
            {"extraction_points": [_point(intervention={"mode": "block_until_signal"})]},
            id="block-without-timeout",
        ),
        pytest.param(
            {"extraction_points": [_point(intervention={"timeout_ms": 50})]},
            id="timeout-without-block",
        ),
        pytest.param(
            {"extraction_points": [_point(intervention={"mode": "block_until_signal", "timeout_ms": 0})]},
            id="timeout-zero",
        ),
        pytest.param(
            {
                "extraction_points": [
                    _point(
                        probe_kind="trajectory",
                        execution_mode="async",
                        intervention={"mode": "block_until_signal", "timeout_ms": 10},
                    )
                ]
            },
            id="async-with-blocking-intervention",
        ),
        pytest.param({"version": 1, "extraction_points": [_point()]}, id="version-not-string"),
        pytest.param({}, id="missing-extraction-points"),
    ],
)
def test_invalid_specs_are_rejected(spec, validator):
    assert not validator.is_valid(spec)
    # The schema only rejects what the parser rejects too.
    assert not _parser_accepts(spec)


@pytest.mark.parametrize(
    "point",
    [
        _point(),
        _point(layers=[0, 4, 8], position=7, probe_args={"threshold": 0.5}),
        _point(position="generated[5:]", probe_kind="trajectory", stride=3, until="fixed_count"),
        _point(position="generated[*]", probe_kind="trajectory", execution_mode="async", queue_depth=4),
        _point(intervention={"mode": "block_until_signal", "timeout_ms": 25, "on_timeout": "abort"}),
        _point(intervention={"mode": "reject"}, stride=None, until=None, queue_depth=None),
        _point(intervention=None),
    ],
)
def test_valid_specs_are_accepted(point, validator):
    spec = {"version": "1", "extraction_points": [point]}
    assert list(validator.iter_errors(spec)) == []
    assert _parser_accepts(spec)


# -- position pattern ---------------------------------------------------------

# Every valid position in tests/spec/test_position.py, plus whitespace the
# parser strips.
VALID_POSITIONS = [
    "7",
    "prompt[0]",
    "prompt[-1]",
    "generated[0]",
    "generated[-1]",
    "generated[*]",
    "generated[5:]",
    "generated[2:8]",
    "generated[5:20]",
    "prompt[-1]+1",
    "generated[5]-2",
    "generated[*]+1",
    " prompt[-1] ",
]

# Invalid string positions from tests/spec/test_position.py, plus typos.
# "generated[5:2]" is missing on purpose: a regex can't compare the bounds,
# so only the parser rejects it.
INVALID_POSITIONS = [
    "prompt[]",
    "generated[]",
    "generated[*",
    "foo[1]",
    "prompt[1.5]",
    "",
    "   ",
    "prompt[*]",
    "prompt[1:3]",
    "generated[-1:]",
    "Prompt[-1]",
    "prompt[-1] + 1",
    "last",
]


def _example_positions():
    for path in EXAMPLE_YAMLS:
        for point in yaml.safe_load(path.read_text(encoding="utf-8"))["extraction_points"]:
            if isinstance(point["position"], str):
                yield point["position"]


@pytest.mark.parametrize("position", VALID_POSITIONS + sorted(set(_example_positions())))
def test_position_pattern_accepts_valid_selectors(position):
    parse_position(position)
    assert re.search(POSITION_PATTERN, position)


@pytest.mark.parametrize("position", INVALID_POSITIONS)
def test_position_pattern_rejects_invalid_selectors(position):
    with pytest.raises(PositionSyntaxError):
        parse_position(position)
    assert not re.search(POSITION_PATTERN, position)


def test_position_description_covers_the_grammar():
    description = json_schema()["$defs"]["ExtractionPointSpec"]["properties"]["position"]["description"]
    for form in ("prompt[i]", "generated[i]", "generated[*]", "generated[start:stop]", "+1"):
        assert form in description

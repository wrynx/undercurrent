"""`undercurrent schema`."""

import json
from pathlib import Path

import pydantic
import pytest
from packaging.version import Version

from undercurrent.cli import main
from undercurrent.spec import json_schema


def test_prints_schema(capsys):
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out) == json_schema()


def test_output_file(tmp_path):
    path = tmp_path / "schema.json"
    assert main(["schema", "-o", str(path)]) == 0
    assert json.loads(path.read_text(encoding="utf-8")) == json_schema()


@pytest.mark.skipif(
    Version(pydantic.VERSION) < Version("2.11"),
    reason="schema/probe-spec.schema.json is generated with pydantic>=2.11; older pydantic "
    "emits an equivalent schema with a different shape (e.g. allOf-wrapped $refs)",
)
def test_output_matches_committed_schema(tmp_path):
    path = tmp_path / "schema.json"
    assert main(["schema", "-o", str(path)]) == 0
    committed = Path(__file__).resolve().parents[2] / "schema" / "probe-spec.schema.json"
    assert path.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8")

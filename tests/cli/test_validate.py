"""`undercurrent validate`: output, exit codes, --strict, --check-probes, JSON."""

import json
import sys
import textwrap

import pytest

from undercurrent.cli import main

VALID = """\
version: "1"
extraction_points:
  - name: a
    layers: 1
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: {probe_type}
    probe_kind: single_shot
  - name: b
    layers: [0, 1]
    tensor_type: mlp_out
    position: "generated[*]"
    probe_type: {probe_type}
    probe_kind: trajectory
"""

# Two independent problems: a negative layer and an unknown tensor_type.
INVALID = """\
version: "1"
extraction_points:
  - name: a
    layers: -1
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: p
    probe_kind: single_shot
  - name: b
    layers: 0
    tensor_type: not_a_tensor
    position: "prompt[-1]"
    probe_type: p
    probe_kind: single_shot
"""

# Uses the deprecated `layer` / `tensor` keys.
ALIASES = """\
version: "1"
extraction_points:
  - name: old
    layer: 3
    tensor: residual_stream
    position: "prompt[-1]"
    probe_type: p
    probe_kind: single_shot
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_valid_file(tmp_path, capsys):
    path = write(tmp_path, "ok.yaml", VALID.format(probe_type="p"))
    assert main(["validate", path]) == 0
    assert capsys.readouterr().out == f"{path}: OK (2 extraction points)\n"


def test_invalid_file_lists_every_issue(tmp_path, capsys):
    path = write(tmp_path, "bad.yaml", INVALID)
    assert main(["validate", path]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"{path}: ERROR 2 issues"
    assert lines[1].startswith("  - line 4: extraction_points[0] 'a'.layers")
    assert lines[2].startswith("  - line 11: extraction_points[1] 'b'.tensor_type")
    assert len(lines) == 3


def test_yaml_syntax_error(tmp_path, capsys):
    path = write(tmp_path, "broken.yaml", "extraction_points: [\n")
    assert main(["validate", path]) == 1
    out = capsys.readouterr().out
    assert f"{path}: ERROR 1 issue" in out
    assert "invalid YAML" in out


def test_missing_file_is_reported_per_file(tmp_path, capsys):
    good = write(tmp_path, "ok.yaml", VALID.format(probe_type="p"))
    missing = str(tmp_path / "missing.yaml")
    assert main(["validate", good, missing]) == 1
    out = capsys.readouterr().out
    assert f"{good}: OK" in out
    assert f"{missing}: ERROR 1 issue\n  - file not found" in out


def test_multiple_files_exit_1_if_any_invalid(tmp_path, capsys):
    good = write(tmp_path, "ok.yaml", VALID.format(probe_type="p"))
    bad = write(tmp_path, "bad.yaml", INVALID)
    assert main(["validate", good, good]) == 0
    assert main(["validate", good, bad]) == 1
    out = capsys.readouterr().out
    assert out.count(f"{good}: OK") == 3
    assert f"{bad}: ERROR" in out


def test_no_files_is_a_usage_error(capsys):
    assert main(["validate"]) == 2
    assert "required" in capsys.readouterr().err


def test_deprecated_keys_warn_but_pass(tmp_path, capsys):
    path = write(tmp_path, "old.yaml", ALIASES)
    assert main(["validate", path]) == 0
    out = capsys.readouterr().out
    assert f"{path}: OK (1 extraction point)" in out
    assert "warning: deprecated spec keys" in out
    assert "rename it to 'layers'" in out


def test_strict_fails_on_deprecated_keys(tmp_path, capsys):
    path = write(tmp_path, "old.yaml", ALIASES)
    assert main(["validate", "--strict", path]) == 1
    out = capsys.readouterr().out
    assert f"{path}: ERROR 1 issue" in out
    assert "deprecated spec keys" in out


def test_strict_passes_canonical_spec(tmp_path):
    assert main(["validate", "--strict", write(tmp_path, "ok.yaml", VALID.format(probe_type="p"))]) == 0


def test_json_output(tmp_path, capsys):
    good = write(tmp_path, "ok.yaml", VALID.format(probe_type="p"))
    bad = write(tmp_path, "bad.yaml", INVALID)
    old = write(tmp_path, "old.yaml", ALIASES)
    assert main(["validate", "--format", "json", good, bad, old]) == 1
    results = json.loads(capsys.readouterr().out)
    assert [r["path"] for r in results] == [good, bad, old]
    assert results[0] == {"path": good, "valid": True, "extraction_points": 2, "errors": [], "warnings": []}
    assert results[1]["valid"] is False
    assert results[1]["extraction_points"] is None
    assert len(results[1]["errors"]) == 2
    assert results[2]["valid"] is True
    assert len(results[2]["warnings"]) == 1


# -- --check-probes ------------------------------------------------------------


@pytest.fixture
def registry():
    return pytest.importorskip("undercurrent.core.registry")


@pytest.fixture
def registered_probe(registry):
    from undercurrent.core import Probe

    class CliTestProbe(Probe):
        probe_kind = "trajectory"

        def on_start(self, request_ctx):
            pass

        def on_activation(self, record):
            return None

        def on_end(self, request_ctx):
            raise NotImplementedError

    registry.register_probe("cli_test_probe", CliTestProbe)
    yield "cli_test_probe"
    registry.unregister_probe("cli_test_probe")


def test_check_probes_hit(tmp_path, capsys, registered_probe):
    path = write(tmp_path, "ok.yaml", VALID.format(probe_type=registered_probe))
    assert main(["validate", "--check-probes", path]) == 0
    assert f"{path}: OK (2 extraction points)" in capsys.readouterr().out


def test_check_probes_miss_suggests_close_name(tmp_path, capsys, registered_probe):
    path = write(tmp_path, "typo.yaml", VALID.format(probe_type="cli_test_prob"))
    assert main(["validate", "--check-probes", path]) == 1
    out = capsys.readouterr().out
    assert f"{path}: ERROR 1 issue" in out  # one issue per probe_type, naming both points
    assert "'a', 'b'" in out
    assert "Did you mean: 'cli_test_probe'?" in out


def test_without_check_probes_unknown_probe_type_is_fine(tmp_path, registry):
    assert main(["validate", write(tmp_path, "ok.yaml", VALID.format(probe_type="nobody_registered_this"))]) == 0


def test_import_registers_probes_for_check(tmp_path, monkeypatch, capsys, registry):
    module = tmp_path / "cli_test_plugin_mod.py"
    module.write_text(
        textwrap.dedent(
            """
            from undercurrent.core import Probe
            from undercurrent.core.registry import register_probe

            @register_probe("cli_plugin_probe")
            class PluginProbe(Probe):
                probe_kind = "trajectory"
                def on_start(self, request_ctx): pass
                def on_activation(self, record): return None
                def on_end(self, request_ctx): raise NotImplementedError
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "cli_test_plugin_mod", raising=False)
    path = write(tmp_path, "ok.yaml", VALID.format(probe_type="cli_plugin_probe"))
    try:
        assert main(["validate", "--check-probes", "--import", "cli_test_plugin_mod", path]) == 0
    finally:
        registry.unregister_probe("cli_plugin_probe")
        sys.modules.pop("cli_test_plugin_mod", None)


def test_import_errors_are_usage_errors(tmp_path, capsys, registry):
    path = write(tmp_path, "ok.yaml", VALID.format(probe_type="p"))
    assert main(["validate", "--check-probes", "--import", "no_such_module_xyz", path]) == 2
    assert capsys.readouterr().err.startswith("error: --import no_such_module_xyz")
    assert main(["validate", "--import", "json", path]) == 2


def test_repo_example_specs_are_valid(capsys):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    specs = sorted(str(p) for p in (root / "examples").rglob("*.yaml"))
    assert specs
    assert main(["validate", "--strict", *specs]) == 0, capsys.readouterr().out

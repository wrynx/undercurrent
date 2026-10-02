"""Top-level `undercurrent` behaviour: help, version, errors, entry points."""

import shutil
import subprocess
import sys

import pytest

from undercurrent import __version__
from undercurrent.cli import _COMMANDS, main


def test_help_lists_every_command(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in _COMMANDS:
        assert name.replace("_", "-") in out
    assert "serve" not in out


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"undercurrent {__version__}"


def test_no_command_prints_help_and_exits_2(capsys):
    assert main([]) == 2
    assert "usage: undercurrent" in capsys.readouterr().err


def test_unknown_command_is_a_usage_error(capsys):
    assert main(["serve"]) == 2
    assert "invalid choice" in capsys.readouterr().err


def test_help_does_not_import_heavy_backends():
    code = (
        "import sys\n"
        "from undercurrent.cli import main\n"
        "for argv in (['--help'], ['inspect-model', '--help'], ['validate', '--help'], ['schema', '--help']):\n"
        "    assert main(argv) == 0\n"
        "print('HEAVY=' + ','.join(m for m in ('torch', 'transformers', 'vllm') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.splitlines()[-1] == "HEAVY=", f"heavy modules imported: {out.stdout.splitlines()[-1]}"


def test_user_errors_print_one_line_without_traceback(tmp_path, capsys):
    missing = tmp_path / "nope"
    assert main(["schema", "-o", str(missing / "schema.json")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err


@pytest.mark.parametrize("position", ["before", "after"])
def test_debug_reraises(tmp_path, position):
    missing = str(tmp_path / "nope" / "schema.json")
    argv = ["--debug", "schema", "-o", missing] if position == "before" else ["schema", "-o", missing, "--debug"]
    with pytest.raises(FileNotFoundError):
        main(argv)


def test_python_dash_m():
    out = subprocess.run([sys.executable, "-m", "undercurrent", "--version"], capture_output=True, text=True)
    assert out.returncode == 0
    assert out.stdout.strip() == f"undercurrent {__version__}"


def test_installed_console_script(tmp_path):
    script = shutil.which("undercurrent")
    if script is None:
        pytest.skip("the `undercurrent` console script isn't on PATH (package not installed)")
    out = subprocess.run([script, "schema"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert '"$schema"' in out.stdout

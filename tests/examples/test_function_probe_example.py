"""examples/function_probe.py runs and registers its probe."""

import runpy
from pathlib import Path

import pytest

from undercurrent.core import default_registry

pytest.importorskip("torch")

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "function_probe.py"


def test_function_probe_example_runs(capsys):
    try:
        runpy.run_path(str(EXAMPLE), run_name="__main__")
        out = capsys.readouterr().out
        assert "score:" in out
        assert "action:" in out
        assert "toxicity" in default_registry
    finally:
        if "toxicity" in default_registry:
            default_registry.unregister("toxicity")

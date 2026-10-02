"""The reference server's command line (no vLLM or GPU needed: `--help` exits before any model loads)."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "openai_server" / "serve_llama_mlp_pipeline.py"


def test_host_defaults_to_localhost():
    out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "default: 127.0.0.1" in out.stdout
    assert "0.0.0.0" in out.stdout  # the container hint

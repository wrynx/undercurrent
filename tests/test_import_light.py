"""Importing the pure-Python subpackages must not pull in the heavy backends."""

from __future__ import annotations

import subprocess
import sys


def test_core_subpackages_do_not_import_heavy_backends():
    code = (
        "import sys\n"
        "import undercurrent.spec, undercurrent.core, undercurrent.router, undercurrent.sinks, "
        "undercurrent.adapters.base\n"
        "print(','.join(m for m in ('torch', 'transformers', 'vllm') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"heavy modules imported: {out.stdout.strip()}"

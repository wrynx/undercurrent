#!/usr/bin/env bash
# Manual GPU check: run the `gpu`-marked tests and the vLLM adapter tests on a
# CUDA machine that already has vLLM installed.
#
# v0.1 has no GPU CI runner, so this is the pre-release GPU gate (see
# RELEASING.md). It runs the same tests as .github/workflows/gpu.yml, against
# whatever vLLM and torch the current environment has. It never installs or
# changes anything.
#
# Setup (the documented production path: vLLM first, then undercurrent into the
# same environment, without the [vllm] extra):
#   pip install "vllm>=0.28,<0.29"      # or use an existing vLLM env or image
#   pip install -e ".[dev]"             # from this checkout
#
# Usage:
#   scripts/gpu_check.sh [--python PY] [--allow-unsupported-vllm] [--live-llama]
#
#   --python PY                Interpreter of the vLLM environment.
#                              Default: python (the active environment).
#   --allow-unsupported-vllm   Run against a vLLM outside SUPPORTED_VLLM (sets
#                              UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM=1). Use it to
#                              evaluate a new vLLM release; a pass does not
#                              widen the supported range by itself.
#   --live-llama               Also run the live Llama content-safety test.
#                              Needs UNDERCURRENT_LIVE_LLAMA_MODEL (a local
#                              Llama-family model path or a cached HF id).
#
# The `gpu` tests include tests/adapters/vllm/test_residual_stream_gpu.py, the
# check that vLLM's `residual_stream` matches the HF backend's on a fused-residual
# (Llama) and a single-tensor (GPT-2) model; the script fails if it didn't run.
#
# Downloads `openai-community/gpt2` (GPT-2) from the Hugging Face Hub (sets RUN_NETWORK_TESTS=1).
# Fails if CUDA or vLLM isn't usable, if the installed vLLM is outside the
# supported range (without --allow-unsupported-vllm), if any test fails, or if
# any GPU test was skipped: a skipped GPU test means it didn't check anything.
# Prints a summary line for docs/compatibility.md at the end.

set -euo pipefail

usage() {
    sed -n '2,/^$/{s/^# \{0,1\}//;p;}' "$0"
}

die() {
    echo "gpu_check: FAIL: $*" >&2
    exit 1
}

step() {
    echo
    echo "==> $*"
}

PY="python"
LIVE_LLAMA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            [[ $# -ge 2 ]] || die "--python needs an interpreter"
            PY="$2"; shift 2 ;;
        --allow-unsupported-vllm)
            export UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM=1; shift ;;
        --live-llama)
            LIVE_LLAMA=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            usage >&2; die "unknown argument: $1" ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v "$PY" >/dev/null 2>&1 || die "interpreter not found: $PY"

if [[ "$LIVE_LLAMA" == 1 && -z "${UNDERCURRENT_LIVE_LLAMA_MODEL:-}" ]]; then
    die "--live-llama needs UNDERCURRENT_LIVE_LLAMA_MODEL set (a local Llama-family model path or a cached HF id)"
fi

export RUN_NETWORK_TESTS=1
export PYTHONUNBUFFERED=1

WORK="$(mktemp -d "${TMPDIR:-/tmp}/undercurrent-gpu-check.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
step "Checking the environment"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

"$PY" -c 'import pytest' 2>/dev/null \
    || die "pytest isn't installed in $PY's environment; run: pip install -e \".[dev]\""

# Without CUDA every GPU test would skip and the check would pass without
# testing anything.
"$PY" - <<'EOF' || die "torch can't see a CUDA GPU (torch.cuda.is_available() is False)"
import sys
import torch
print("torch", torch.__version__, "CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit(1)
print("GPU:", torch.cuda.get_device_name(0))
EOF

"$PY" -c 'import vllm; print("vllm", vllm.__version__)' \
    || die "vllm isn't importable in $PY's environment; install vLLM first, then undercurrent into the same environment"

# The tests import undercurrent; make sure it's the code in this checkout.
"$PY" - "$REPO_ROOT" <<'EOF' || die "undercurrent isn't installed from this checkout; run: pip install -e \".[dev]\""
import os, sys
import undercurrent
root = os.path.realpath(os.path.join(sys.argv[1], "src", "undercurrent"))
here = os.path.realpath(os.path.dirname(undercurrent.__file__))
print("undercurrent", undercurrent.__version__, "from", here)
if here != root:
    print(f"expected {root}", file=sys.stderr)
    sys.exit(1)
EOF

# Same check the adapter runs at construction. With --allow-unsupported-vllm
# it warns instead of raising.
"$PY" -c 'from undercurrent.adapters.vllm.version_check import SUPPORTED_VLLM, check_vllm_version; print("vllm", check_vllm_version(), "checked against", SUPPORTED_VLLM)' \
    || die "the installed vLLM is outside the supported range; use a vLLM in range, or pass --allow-unsupported-vllm to evaluate it anyway"

# ---------------------------------------------------------------------------
step "pytest -m gpu"

# The live Llama test only runs on request (below).
status=0
"$PY" -m pytest -v -rs -m gpu \
    --ignore=tests/examples/content_safety/test_live_llama_integration.py \
    --junitxml="$WORK/gpu.xml" || status=1

# A GPU test that skipped (missing vLLM, no network opt-in, ...) checked
# nothing, so treat it as a failure.
"$PY" - "$WORK/gpu.xml" <<'EOF' || status=1
import sys
import xml.etree.ElementTree as ET

try:
    root = ET.parse(sys.argv[1]).getroot()
except (OSError, ET.ParseError) as exc:
    print(f"gpu_check: no pytest report ({exc})", file=sys.stderr)
    sys.exit(1)
cases = list(root.iter("testcase"))
skipped = [c for c in cases if c.find("skipped") is not None]
for c in skipped:
    print(f"gpu_check: skipped: {c.get('classname')}::{c.get('name')}: {c.find('skipped').get('message')}", file=sys.stderr)
if not cases:
    print("gpu_check: no gpu-marked tests were collected", file=sys.stderr)
    sys.exit(1)
if skipped:
    print(f"gpu_check: {len(skipped)} of {len(cases)} GPU tests were skipped", file=sys.stderr)
    sys.exit(1)
# The residual_stream HF-vs-vLLM equivalence test is the GPU verification of
# the 0.1.0 fused-residual fix; make sure it wasn't deselected or renamed away.
if not any("test_residual_stream_gpu" in (c.get("classname") or "") for c in cases):
    print("gpu_check: tests/adapters/vllm/test_residual_stream_gpu.py didn't run", file=sys.stderr)
    sys.exit(1)
EOF

# ---------------------------------------------------------------------------
step "pytest tests/adapters/vllm (non-GPU tests, against real vLLM)"

"$PY" -m pytest -v -rs -m "not gpu" tests/adapters/vllm || status=1

# ---------------------------------------------------------------------------
if [[ "$LIVE_LLAMA" == 1 ]]; then
    step "Live Llama content-safety test"
    "$PY" -m pytest -v -rs tests/examples/content_safety/test_live_llama_integration.py || status=1
fi

# ---------------------------------------------------------------------------
step "Summary"

summary="$("$PY" - <<'EOF'
import platform
from importlib.metadata import version
import torch
print(
    f"Python {platform.python_version()} | torch {version('torch')} | "
    f"transformers {version('transformers')} | vllm {version('vllm')} | "
    f"CUDA {torch.version.cuda} | GPU {torch.cuda.get_device_name(0)}"
)
EOF
)"
echo "$summary"
if [[ -n "${UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM:-}" ]]; then
    echo "(ran with UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM=1)"
fi

if [[ "$status" != 0 ]]; then
    die "see the output above"
fi
echo "gpu_check: OK"

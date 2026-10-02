#!/usr/bin/env bash
# Smoke-test the built undercurrent artefacts (wheel + sdist), not the source tree.
#
# Catches packaging bugs an editable install hides: files missing from the
# wheel, a wrong package list, demo code leaking into the install, broken
# entry points, imports that only work from a checkout.
#
# Usage:
#   scripts/smoke_test_wheel.sh [--dist DIR] [--install wheel|sdist] [--python PY]
#                               [--site-packages-from DIR]
#
#   --dist DIR                 Test the wheel and sdist already in DIR (exactly one
#                              of each). Default: build both into a temp dir with
#                              `python -m build` (installed into a temp venv).
#   --install wheel|sdist      Which artefact to pip-install for the installed-package
#                              checks (both are always content-checked). Installing
#                              the sdist builds a wheel from it, so it needs network
#                              for the build backend. Default: wheel.
#   --python PY                Interpreter used to create the temp venvs.
#                              Default: python3.
#   --site-packages-from DIR   Give the fresh install venv a .pth pointing at DIR, so
#                              the heavy base dependencies (torch, transformers) come
#                              from an existing environment instead of a download.
#                              DIR must belong to the same Python version as PY.
#                              Packages installed into the fresh venv still win.
#
# Exits non-zero on any failure. Temp dirs are removed on exit.

set -euo pipefail

usage() {
    sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"
}

die() {
    echo "smoke_test_wheel: FAIL: $*" >&2
    exit 1
}

step() {
    echo
    echo "==> $*"
}

DIST_DIR=""
INSTALL_FROM="wheel"
PY="python3"
SITE_FROM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dist)
            [[ $# -ge 2 ]] || die "--dist needs a directory"
            DIST_DIR="$2"; shift 2 ;;
        --install)
            [[ $# -ge 2 ]] || die "--install needs wheel or sdist"
            INSTALL_FROM="$2"; shift 2 ;;
        --python)
            [[ $# -ge 2 ]] || die "--python needs an interpreter"
            PY="$2"; shift 2 ;;
        --site-packages-from)
            [[ $# -ge 2 ]] || die "--site-packages-from needs a directory"
            SITE_FROM="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            usage >&2; die "unknown argument: $1" ;;
    esac
done

[[ "$INSTALL_FROM" == wheel || "$INSTALL_FROM" == sdist ]] || die "--install must be wheel or sdist, got: $INSTALL_FROM"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v "$PY" >/dev/null 2>&1 || die "interpreter not found: $PY"
PY_VERSION="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

if [[ -n "$SITE_FROM" ]]; then
    [[ -d "$SITE_FROM" ]] || die "--site-packages-from: not a directory: $SITE_FROM"
    SITE_FROM="$(cd "$SITE_FROM" && pwd)"
    if [[ "$SITE_FROM" =~ /python([0-9]+\.[0-9]+)/ && "${BASH_REMATCH[1]}" != "$PY_VERSION" ]]; then
        die "--site-packages-from is for Python ${BASH_REMATCH[1]} but $PY is Python $PY_VERSION (use --python)"
    fi
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/undercurrent-smoke.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

PIP_FLAGS=(--disable-pip-version-check --no-input -q)

# Run Python in an isolated environment: no user site, no PYTHONPATH, no
# downloads, and a cwd outside the checkout so nothing resolves from source.
mkdir -p "$WORK/run"
isolated() {
    (cd "$WORK/run" && env -u PYTHONPATH PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$@")
}

# ---------------------------------------------------------------------------
# 1. Build (unless --dist was given)
# ---------------------------------------------------------------------------
if [[ -z "$DIST_DIR" ]]; then
    step "Building wheel and sdist from $REPO_ROOT"
    "$PY" -m venv "$WORK/build-venv"
    "$WORK/build-venv/bin/python" -m pip install "${PIP_FLAGS[@]}" build
    DIST_DIR="$WORK/dist"
    "$WORK/build-venv/bin/python" -m build --outdir "$DIST_DIR" "$REPO_ROOT" >"$WORK/build.log" 2>&1 \
        || { cat "$WORK/build.log" >&2; die "python -m build failed"; }
fi

[[ -d "$DIST_DIR" ]] || die "--dist: not a directory: $DIST_DIR"
DIST_DIR="$(cd "$DIST_DIR" && pwd)"
shopt -s nullglob
wheels=("$DIST_DIR"/undercurrent-*.whl)
sdists=("$DIST_DIR"/undercurrent-*.tar.gz)
shopt -u nullglob
[[ ${#wheels[@]} -eq 1 ]] || die "expected exactly one undercurrent-*.whl in $DIST_DIR, found ${#wheels[@]}"
[[ ${#sdists[@]} -eq 1 ]] || die "expected exactly one undercurrent-*.tar.gz in $DIST_DIR, found ${#sdists[@]}"
WHEEL="${wheels[0]}"
SDIST="${sdists[0]}"
echo "wheel: $WHEEL"
echo "sdist: $SDIST"

# ---------------------------------------------------------------------------
# 2. Artefact checks (no install)
# ---------------------------------------------------------------------------
step "Checking artefact contents"
isolated "$PY" - "$WHEEL" "$SDIST" <<'PYEOF' || die "artefact checks failed"
import sys
import tarfile
import zipfile

wheel_path, sdist_path = sys.argv[1], sys.argv[2]
errors = []

with zipfile.ZipFile(wheel_path) as zf:
    names = zf.namelist()
top_level = {n.split("/", 1)[0] for n in names}
dist_info = [t for t in top_level if t.endswith(".dist-info")]

if "undercurrent/__init__.py" not in names:
    errors.append("wheel is missing undercurrent/__init__.py")
for lic in ("LICENSE", "NOTICE"):
    if not any(n.startswith(tuple(d + "/" for d in dist_info)) and n.rsplit("/", 1)[-1] == lic for n in names):
        errors.append(f"wheel is missing license file {lic} under *.dist-info/")
if "undercurrent/py.typed" not in names:
    print("WARNING: wheel has no undercurrent/py.typed (expected once p4-typing lands)")

for entry in sorted(top_level):
    if entry in ("tests", "examples") or entry.startswith("probing_"):
        errors.append(f"wheel contains forbidden top-level entry {entry!r}")
    elif entry != "undercurrent" and not (
        entry.startswith("undercurrent-") and entry.endswith((".dist-info", ".data"))
    ):
        errors.append(f"wheel contains unexpected top-level entry {entry!r}")
if len(dist_info) != 1:
    errors.append(f"wheel should have exactly one .dist-info dir, found {dist_info}")

with tarfile.open(sdist_path) as tf:
    snames = tf.getnames()
roots = {n.split("/", 1)[0] for n in snames}
if len(roots) != 1:
    errors.append(f"sdist should have one root dir, found {sorted(roots)}")
else:
    root = roots.pop()
    rel = {n[len(root) + 1:] for n in snames}
    for required in ("LICENSE", "pyproject.toml"):
        if required not in rel:
            errors.append(f"sdist is missing {required}")
    if not any(r.startswith("tests/") for r in rel):
        errors.append("sdist is missing tests/")

print(f"wheel: {len(names)} files, top level {sorted(top_level)}")
print(f"sdist: {len(snames)} entries")
if errors:
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
print("artefact checks OK")
PYEOF

# ---------------------------------------------------------------------------
# 3. Base install into a fresh venv (no extras)
# ---------------------------------------------------------------------------
if [[ "$INSTALL_FROM" == sdist ]]; then ARTEFACT="$SDIST"; else ARTEFACT="$WHEEL"; fi
step "Installing the $INSTALL_FROM (no extras) into a fresh venv"
VENV="$WORK/venv"
"$PY" -m venv "$VENV"
VPY="$VENV/bin/python"
if [[ -n "$SITE_FROM" ]]; then
    PURELIB="$("$VPY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    echo "$SITE_FROM" >"$PURELIB/zz_smoke_base_site.pth"
    echo "base packages from: $SITE_FROM"
fi
isolated "$VPY" -m pip install "${PIP_FLAGS[@]}" "$ARTEFACT" || die "pip install of the $INSTALL_FROM failed"

step "Running installed-package checks"
cat >"$WORK/run/smoke_checks.py" <<'PYEOF'
import importlib
import importlib.metadata as md
import os
import pkgutil
import subprocess
import sys
import sysconfig
import traceback

PURELIB = os.path.realpath(sysconfig.get_paths()["purelib"])
failures = []


def check(label, fn):
    print(f"--- {label}")
    try:
        fn()
    except Exception:
        traceback.print_exc()
        failures.append(label)


def under_venv(path):
    return os.path.realpath(path).startswith(PURELIB + os.sep)


# -- the package itself -------------------------------------------------------
import undercurrent

print(f"undercurrent imported from {undercurrent.__file__}")
if not under_venv(undercurrent.__file__):
    sys.exit(f"ERROR: undercurrent imported from {undercurrent.__file__}, not the venv ({PURELIB})")


def import_all():
    skipped, broken = [], []
    names = []

    def onerror(name):
        broken.append((name, "".join(traceback.format_exception(*sys.exc_info()))))

    for info in pkgutil.walk_packages(undercurrent.__path__, "undercurrent.", onerror=onerror):
        names.append(info.name)
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as exc:
            # Only a missing vLLM is an acceptable reason to skip a module.
            if exc.name and exc.name.split(".")[0] == "vllm":
                skipped.append(info.name)
            else:
                broken.append((info.name, traceback.format_exc()))
        except Exception:
            broken.append((info.name, traceback.format_exc()))
    print(f"imported {len(names) - len(skipped) - len(broken)}/{len(names)} modules")
    for name in skipped:
        print(f"WARNING: skipped {name} (needs vLLM at import time; vLLM imports should be lazy)")
    for name, tb in broken:
        print(f"ERROR importing {name}:\n{tb}", file=sys.stderr)
    assert not broken, f"{len(broken)} module(s) failed to import"

    # walk_packages silently skips directories without __init__.py, so cross-check
    # against every .py file the wheel actually installed.
    files = md.distribution("undercurrent").files or []
    installed = set()
    for f in files:
        parts = f.parts
        if parts[0] == "undercurrent" and f.suffix == ".py":
            mod = ".".join(parts[:-1] if parts[-1] == "__init__.py" else parts[:-1] + (f.stem,))
            installed.add(mod)
    unreached = sorted(installed - set(names) - {"undercurrent"})
    assert not unreached, f"installed modules not reachable as regular packages (missing __init__.py?): {unreached}"
    assert "undercurrent.adapters.hf" in names and "undercurrent.adapters.vllm" in names, names
    assert "vllm" not in sys.modules, "importing undercurrent pulled in vllm eagerly"


def no_demo_code():
    for mod in ("examples", "content_safety_demo"):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        where = getattr(m, "__file__", None) or list(getattr(m, "__path__", []))
        # Something unrelated in a shared base env is not our leak; something
        # in this venv's site-packages is.
        paths = [where] if isinstance(where, str) else where
        assert not any(under_venv(p) for p in paths), f"demo package {mod!r} leaked into the install: {where}"
        print(f"note: {mod!r} importable from outside this venv ({where}); not from the wheel")
    print("no demo code installed")


def version():
    v = undercurrent.__version__
    print(f"undercurrent.__version__ = {v}")
    assert v == md.version("undercurrent"), (v, md.version("undercurrent"))
    assert "unknown" not in v, v


SPEC_YAML = """
version: "1"
extraction_points:
  - name: last_prompt
    layers: 0
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: mlp_classifier
    probe_kind: single_shot
    execution_mode: inline
  - name: trajectory
    layers: [0, 1]
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: trajectory_score
    probe_kind: trajectory
    execution_mode: async
    queue_depth: 8
"""


def synthetic_pipeline():
    from undercurrent.core import RequestContext
    from undercurrent.core.examples import MLPClassifierProbe, TrajectoryScoreProbe
    from undercurrent.router import ProbeFactory, Router
    from undercurrent.sinks import FileLogSink, wire_router
    from undercurrent.spec import ActivationRecord, parse_yaml

    spec = parse_yaml(SPEC_YAML)
    points = list(spec)
    assert [p.name for p in points] == ["last_prompt", "trajectory"], [p.name for p in points]

    router = Router(
        probe_registry={
            "mlp_classifier": ProbeFactory(MLPClassifierProbe, {"num_classes": 3}),
            "trajectory_score": ProbeFactory(TrajectoryScoreProbe, {"threshold": 1e9}),
        }
    )
    log_path = os.path.join(os.getcwd(), "observations.ndjson")
    wire_router(router, FileLogSink(log_path))
    try:
        rid = "smoke-req"
        router.register_request(rid, points, RequestContext(rid, {"prompt_len": 4}, None))
        signal = router.route(ActivationRecord(rid, "last_prompt", 0, 3, "residual_stream", [0.5, 1.5], False))
        assert signal is not None, "inline probe returned no signal"
        for pos in range(4, 8):
            for layer in (0, 1):
                router.route(ActivationRecord(rid, "trajectory", layer, pos, "residual_stream", [0.1, 0.2], True))
        results = router.end_request(rid)
    finally:
        router.shutdown()

    assert set(results) == {"last_prompt", "trajectory"}, set(results)
    assert results["last_prompt"].verdict["predicted_class"] == 2, results["last_prompt"].verdict
    assert results["trajectory"].verdict["count"] == 8, results["trajectory"].verdict
    with open(log_path) as fh:
        logged = [line for line in fh if line.strip()]
    assert logged, "FileLogSink wrote nothing for the async binding"
    print(f"results: { {k: r.verdict for k, r in results.items()} }")


def hf_cpu_generation():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from undercurrent.adapters.hf import HFEngineAdapter
    from undercurrent.core.examples import TrajectoryScoreProbe
    from undercurrent.router import ProbeFactory, Router
    from undercurrent.spec import parse_yaml

    vocab_size = 64
    config = GPT2Config(
        n_layer=2, n_embd=32, n_head=2, n_positions=64, n_inner=64,
        vocab_size=vocab_size, bos_token_id=0, eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    backend = Tokenizer(models.WordLevel(vocab={f"tok{i}": i for i in range(vocab_size)}, unk_token="tok0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="tok0")

    points = list(parse_yaml("""
version: "1"
extraction_points:
  - name: gen
    layers: [0, 1]
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: trajectory_score
    probe_kind: trajectory
    execution_mode: inline
"""))
    router = Router(probe_registry={"trajectory_score": ProbeFactory(TrajectoryScoreProbe, {"threshold": 1e9})})
    captured = {}
    original_end = router.end_request

    def spy(request_id):
        captured["results"] = original_end(request_id)
        return captured["results"]

    router.end_request = spy

    adapter = HFEngineAdapter()
    adapter.load_model(model, tokenizer=tokenizer, device="cpu")
    try:
        adapter.register_extraction("hf-req", points)
        text = adapter.generate(
            "hf-req", "tok1 tok2 tok3", {"max_new_tokens": 4, "min_new_tokens": 4, "do_sample": False}, router
        )
    finally:
        router.shutdown()

    verdict = captured["results"]["gen"].verdict
    print(f"generated {text!r}; verdict {verdict}")
    assert verdict["count"] > 0, verdict


def console_scripts():
    dist = md.distribution("undercurrent")
    eps = [ep for ep in dist.entry_points if ep.group == "console_scripts"]
    print(f"{len(eps)} console script(s): {[ep.name for ep in eps]}")
    for ep in eps:
        exe = os.path.join(sys.prefix, "bin", ep.name)
        assert os.path.exists(exe), f"console script {ep.name} not installed at {exe}"
        proc = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, f"`{ep.name} --help` exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
        print(f"`{ep.name} --help` OK")


def vllm_plugin_declared():
    dist = md.distribution("undercurrent")
    assert under_venv(str(dist.locate_file("undercurrent/__init__.py"))), "metadata not from the venv"
    eps = [ep for ep in dist.entry_points if ep.group == "vllm.general_plugins"]
    assert eps, "no vllm.general_plugins entry point declared"
    for ep in eps:
        print(f"vllm.general_plugins: {ep.name} = {ep.value}")
        # Resolve the target without calling it or importing vLLM.
        assert callable(ep.load()), ep.value
    assert "vllm" not in sys.modules, "resolving the vLLM plugin entry point imported vllm"


check("import every undercurrent module", import_all)
check("no demo code in the install", no_demo_code)
check("version", version)
check("synthetic router pipeline", synthetic_pipeline)
check("HF CPU generation (random tiny GPT-2)", hf_cpu_generation)
check("console scripts --help", console_scripts)
check("vLLM plugin entry point declared", vllm_plugin_declared)

if failures:
    print(f"\nFAILED checks: {failures}", file=sys.stderr)
    sys.exit(1)
print("\nall installed-package checks OK")
PYEOF
isolated "$VPY" smoke_checks.py || die "installed-package checks failed"

echo
echo "smoke_test_wheel: OK ($(basename "$ARTEFACT"))"

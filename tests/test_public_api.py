"""Pin Undercurrent's public API.

`PUBLIC_API` is the snapshot of ``sorted(module.__all__)`` for the top-level
package and every public subpackage. If a test here fails, the public surface
changed. If that was deliberate:

1. update `PUBLIC_API` below to match,
2. add the change to CHANGELOG.md (a removal or rename is a breaking change;
   see docs/api-stability.md for the deprecation policy),
3. make sure each new name has a docstring.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import sys
import typing
from pathlib import Path

import pytest

import undercurrent

PUBLIC_API: dict[str, list[str]] = {
    "undercurrent": [
        "ActivationRecord",
        "EngineAdapter",
        "ExecutionMode",
        "ExtractionPoint",
        "FileLogSink",
        "GenerationOutput",
        "InMemoryMetricsRegistry",
        "InterventionMode",
        "InterventionPolicy",
        "LogSink",
        "MetricsSink",
        "MetricsSnapshot",
        "MissingDependencyError",
        "OverflowPolicy",
        "Probe",
        "ProbeAction",
        "ProbeFactory",
        "ProbeKind",
        "ProbeNotFoundError",
        "ProbeResult",
        "ProbeSignal",
        "ProbeSpec",
        "ProbedModel",
        "ProbingError",
        "RequestContext",
        "RequestHandle",
        "Router",
        "RouterError",
        "SpecValidationError",
        "TensorType",
        "TimeoutAction",
        "WebhookLogSink",
        "__version__",
        "chain",
        "drop_keys",
        "load_spec",
        "parse_position",
        "probe",
        "redact_keys",
        "register_probe",
    ],
    "undercurrent.adapters": [
        "EngineAdapter",
        "MissingDependencyError",
    ],
    "undercurrent.adapters.hf": [
        "HFAdapterLimitationError",
        "HFEngineAdapter",
    ],
    "undercurrent.adapters.vllm": [
        "VLLMAdapterLimitationError",
        "VLLMEngineAdapter",
    ],
    "undercurrent.cli": [
        "main",
    ],
    "undercurrent.core": [
        "FunctionProbe",
        "Probe",
        "ProbeAction",
        "ProbeFactory",
        "ProbeNotFoundError",
        "ProbeRegistry",
        "ProbeResult",
        "ProbeSignal",
        "RequestContext",
        "default_registry",
        "get_probe_factory",
        "list_probes",
        "probe",
        "register_probe",
        "unregister_probe",
    ],
    "undercurrent.core.examples": [
        "MLPClassifierProbe",
        "TrajectoryScoreProbe",
    ],
    "undercurrent.errors": [
        "ProbeDefinitionError",
        "ProbingError",
        "ProbingKeyError",
        "ProbingRuntimeError",
        "ProbingTypeError",
        "ProbingValueError",
        "SpecFileNotFoundError",
    ],
    "undercurrent.model": [
        "DEFAULT_MAX_NEW_TOKENS",
        "DEFAULT_VLLM_MAX_CONCURRENCY",
        "GenerationOutput",
        "ProbedModel",
        "ProbedModelConfigError",
    ],
    "undercurrent.router": [
        "DEFAULT_CIRCUIT_BREAKER_THRESHOLD",
        "DEFAULT_DRAIN_TIMEOUT",
        "DEFAULT_QUEUE_DEPTH",
        "InMemoryMetricsRegistry",
        "MetricsSink",
        "MetricsSnapshot",
        "OverflowPolicy",
        "RequestEndListener",
        "RequestHandle",
        "Router",
        "RouterError",
        "default_worker_pool_size",
    ],
    "undercurrent.sinks": [
        "DEFAULT_PROMPT_TEXT_KEYS",
        "FileLogSink",
        "LogSink",
        "RedactFn",
        "WebhookLogSink",
        "chain",
        "drop_keys",
        "redact_keys",
        "to_jsonable",
        "wire_router",
    ],
    "undercurrent.spec": [
        "ActivationRecord",
        "ExecutionMode",
        "ExtractionPoint",
        "FrozenArgs",
        "InterventionMode",
        "InterventionPolicy",
        "PositionKind",
        "PositionSelector",
        "PositionSyntaxError",
        "ProbeKind",
        "ProbeSpec",
        "ProbingSpecError",
        "SpecValidationError",
        "TensorType",
        "TimeoutAction",
        "UntilKind",
        "extraction_point_to_dict",
        "json_schema",
        "load_spec",
        "load_yaml_file",
        "parse_dict",
        "parse_position",
        "parse_yaml",
        "probe_spec_to_dict",
        "to_yaml",
    ],
}

#: The front door: listed first in `undercurrent.__all__`, in this order.
FRONT_DOOR = [
    "ProbedModel",
    "GenerationOutput",
    "probe",
    "register_probe",
    "Probe",
    "ExtractionPoint",
    "load_spec",
    "ProbingError",
    "__version__",
]

HOW_TO_UPDATE = (
    "The public API of {module} changed.\n"
    "  added:   {added}\n"
    "  removed: {removed}\n"
    "If this is deliberate, update PUBLIC_API in tests/test_public_api.py and record the change in "
    "CHANGELOG.md (removals and renames are breaking; follow the deprecation policy in "
    "docs/api-stability.md). If it isn't, fix the module's __all__."
)

HEAVY = ("torch", "transformers", "vllm")


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.mark.parametrize("module_name", sorted(PUBLIC_API))
def test_public_api_is_pinned(module_name):
    actual = sorted(importlib.import_module(module_name).__all__)
    expected = PUBLIC_API[module_name]
    assert actual == expected, HOW_TO_UPDATE.format(
        module=module_name,
        added=sorted(set(actual) - set(expected)) or "-",
        removed=sorted(set(expected) - set(actual)) or "-",
    )


@pytest.mark.parametrize("module_name", sorted(PUBLIC_API))
def test_all_has_no_duplicates(module_name):
    names = importlib.import_module(module_name).__all__
    assert len(names) == len(set(names))


def test_every_public_subpackage_is_pinned():
    """A new subpackage with an ``__all__`` must be added to PUBLIC_API (or made private with a leading ``_``)."""
    root = Path(undercurrent.__file__).parent
    found = set()
    for init in root.rglob("__init__.py"):
        parts = init.parent.relative_to(root.parent).parts
        if any(p.startswith("_") for p in parts):
            continue
        found.add(".".join(parts))
    assert found <= set(PUBLIC_API), f"subpackages missing from PUBLIC_API: {sorted(found - set(PUBLIC_API))}"


def test_front_door_comes_first():
    assert undercurrent.__all__[: len(FRONT_DOOR)] == FRONT_DOOR


@pytest.mark.parametrize(("module_name", "name"), [(m, n) for m, names in PUBLIC_API.items() for n in names])
def test_every_name_resolves(module_name, name):
    module = importlib.import_module(module_name)
    assert getattr(module, name) is not None


def _is_documentable(obj: object) -> bool:
    # On Python 3.10, inspect.isclass() is True for generic aliases such as
    # `Callable[[str], None]`; those are type aliases, not classes.
    if typing.get_origin(obj) is not None:
        return False
    return inspect.isclass(obj) or inspect.isfunction(obj)


def _own_docstring(obj: object) -> str:
    doc = (obj.__dict__.get("__doc__") if inspect.isclass(obj) else obj.__doc__) or ""
    # dataclasses generate "Name(field: type, ...)" when there's no docstring.
    if inspect.isclass(obj) and doc.startswith(f"{obj.__name__}("):
        return ""
    return doc.strip()


@pytest.mark.parametrize(("module_name", "name"), [(m, n) for m, names in PUBLIC_API.items() for n in names])
def test_every_public_callable_has_a_docstring(module_name, name):
    obj = getattr(importlib.import_module(module_name), name)
    if not _is_documentable(obj):
        pytest.skip(f"{name} is a constant or type alias")
    assert _own_docstring(obj), f"{module_name}.{name} is public but has no docstring"


@pytest.mark.parametrize("module_name", sorted(PUBLIC_API))
def test_every_public_module_has_a_docstring(module_name):
    assert (importlib.import_module(module_name).__doc__ or "").strip()


# ---------------------------------------------------------------------------
# The top-level package
# ---------------------------------------------------------------------------


def test_top_level_names_are_the_canonical_objects():
    for name in undercurrent.__all__:
        if name == "__version__":
            continue
        home = importlib.import_module(undercurrent._LAZY[name], "undercurrent")
        assert getattr(undercurrent, name) is getattr(home, name), name


def test_lazy_table_matches_all():
    assert set(undercurrent._LAZY) | {"__version__"} == set(undercurrent.__all__)


def test_type_checking_block_imports_every_lazy_name():
    """IDEs and type checkers only see the TYPE_CHECKING imports; keep them in step with `_LAZY`."""
    tree = ast.parse(Path(undercurrent.__file__).read_text(encoding="utf-8"))
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
    )
    imported = {}
    for node in block.body:
        assert isinstance(node, ast.ImportFrom)
        for alias in node.names:
            imported[alias.asname or alias.name] = "." * node.level + (node.module or "")
    assert imported == undercurrent._LAZY


def test_star_import_exposes_exactly_all():
    namespace: dict[str, object] = {}
    exec("from undercurrent import *", namespace)
    namespace.pop("__builtins__")
    assert sorted(namespace) == sorted(undercurrent.__all__)


def test_dir_lists_every_public_name():
    assert set(undercurrent.__all__) <= set(dir(undercurrent))


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'NotAThing'"):
        undercurrent.NotAThing


def test_import_is_light():
    """``import undercurrent`` loads no subpackage, no heavy backend and not even pydantic/yaml."""
    loaded = _run(
        "import sys\n"
        "import undercurrent\n"
        "print(','.join(sorted(m for m in sys.modules if m.startswith('undercurrent.') or m in "
        "('torch', 'transformers', 'vllm', 'pydantic', 'yaml', 'numpy'))))\n"
    )
    assert loaded in ("", "undercurrent._version"), f"import undercurrent loaded: {loaded}"


def test_front_door_access_does_not_import_heavy_backends():
    """Touching every top-level name (what ``from undercurrent import *`` does) stays torch-free."""
    loaded = _run(
        f"import sys\nfrom undercurrent import *\nprint(','.join(m for m in {HEAVY!r} if m in sys.modules))\n"
    )
    assert loaded == "", f"heavy modules imported: {loaded}"


def test_vllm_adapter_package_does_not_import_vllm():
    loaded = _run(
        f"import sys\nimport undercurrent.adapters.vllm\nprint(','.join(m for m in {HEAVY!r} if m in sys.modules))\n"
    )
    assert loaded == "", f"heavy modules imported: {loaded}"


# ---------------------------------------------------------------------------
# load_spec
# ---------------------------------------------------------------------------

SPEC_YAML = """\
extraction_points:
  - name: last
    layers: 0
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: norm
    probe_kind: single_shot
"""


def test_load_spec_accepts_every_form(tmp_path):
    from undercurrent import ProbeSpec, load_spec
    from undercurrent.spec import parse_yaml

    path = tmp_path / "probes.yaml"
    path.write_text(SPEC_YAML, encoding="utf-8")
    expected = parse_yaml(SPEC_YAML)

    import yaml

    for source in (path, str(path), SPEC_YAML, yaml.safe_load(SPEC_YAML)):
        assert load_spec(source) == expected
    assert load_spec(expected) is expected
    assert isinstance(expected, ProbeSpec)


def test_load_spec_missing_file_is_a_clear_error(tmp_path):
    from undercurrent import ProbingError, load_spec
    from undercurrent.errors import SpecFileNotFoundError

    with pytest.raises(SpecFileNotFoundError, match="not found"):
        load_spec(str(tmp_path / "nope.yaml"))
    with pytest.raises(ProbingError):
        load_spec(tmp_path / "nope.yaml")


def test_load_spec_rejects_other_types():
    from undercurrent import ProbingError, load_spec

    with pytest.raises(TypeError, match="load_spec\\(\\) takes"):
        load_spec(42)  # type: ignore[arg-type]
    with pytest.raises(ProbingError):
        load_spec([1, 2])  # type: ignore[arg-type]

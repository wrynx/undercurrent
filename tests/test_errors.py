"""`undercurrent.errors.ProbingError`: one base for every library error, and
actionable messages for the most common mistakes.

The golden tests assert substrings (what went wrong, where, how to fix it),
never whole messages, so wording can still be polished.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings

import pytest

import undercurrent
from undercurrent.errors import (
    ProbeDefinitionError,
    ProbingError,
    ProbingKeyError,
    ProbingRuntimeError,
    ProbingTypeError,
    ProbingValueError,
    SpecFileNotFoundError,
)

GOOD_POINT = """\
  - name: toxicity
    layers: [1]
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: err_test_probe
    probe_kind: single_shot
"""


def spec_with(**overrides: str) -> str:
    """A one-point YAML spec with some lines replaced (``key=None`` drops a key)."""
    lines = ["extraction_points:"]
    for line in GOOD_POINT.splitlines():
        key = line.strip().lstrip("- ").split(":")[0]
        if key in overrides:
            value = overrides.pop(key)
            if value is None:
                continue
            prefix = "  - " if line.lstrip().startswith("- ") else "    "
            line = f"{prefix}{value}"
        lines.append(line)
    lines += [f"    {extra}" for extra in overrides.values()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def _library_exception_classes() -> list[type[BaseException]]:
    """Every exception class defined in a module of the installed package."""
    found: dict[str, type[BaseException]] = {}
    for info in pkgutil.walk_packages(undercurrent.__path__, "undercurrent."):
        if info.name.endswith("__main__"):
            continue
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseException)
                and obj.__module__.startswith("undercurrent.")
                and not obj.__module__.startswith("undercurrent.core.examples")
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return [found[k] for k in sorted(found)]


LIBRARY_EXCEPTIONS = _library_exception_classes()


def test_found_the_library_exceptions():
    names = {cls.__name__ for cls in LIBRARY_EXCEPTIONS}
    assert {
        "ProbingSpecError",
        "SpecValidationError",
        "PositionSyntaxError",
        "RouterError",
        "ProbeNotFoundError",
        "ProbedModelConfigError",
        "HFAdapterLimitationError",
        "VLLMAdapterLimitationError",
        "VLLMIntrospectionError",
        "SeqMapperError",
        "WorkerExtractionError",
        "MissingDependencyError",
        "CLIError",
        "UsageError",
    } <= names


@pytest.mark.parametrize("cls", LIBRARY_EXCEPTIONS, ids=lambda c: c.__name__)
def test_every_library_exception_is_a_probing_error(cls):
    assert issubclass(cls, ProbingError)


def test_probing_error_module_imports_nothing_from_the_package():
    import undercurrent.errors as errors_mod

    source = inspect.getsource(errors_mod)
    assert "from ." not in source and "import undercurrent" not in source


@pytest.mark.parametrize(
    ("dotted", "old_base"),
    [
        ("undercurrent.core.registry.ProbeNotFoundError", KeyError),
        ("undercurrent.model.ProbedModelConfigError", ValueError),
        ("undercurrent.adapters._optional.MissingDependencyError", ImportError),
        ("undercurrent.errors.ProbingValueError", ValueError),
        ("undercurrent.errors.ProbingTypeError", TypeError),
        ("undercurrent.errors.ProbingRuntimeError", RuntimeError),
        ("undercurrent.errors.ProbingKeyError", KeyError),
        ("undercurrent.errors.ProbeDefinitionError", TypeError),
        ("undercurrent.errors.SpecFileNotFoundError", FileNotFoundError),
        ("undercurrent.spec.errors.SpecValidationError", Exception),
        ("undercurrent.router.errors.RouterError", Exception),
        ("undercurrent.cli._errors.UsageError", Exception),
    ],
)
def test_old_bases_are_preserved(dotted, old_base):
    module_name, _, name = dotted.rpartition(".")
    cls = getattr(importlib.import_module(module_name), name)
    assert issubclass(cls, old_base)
    assert issubclass(cls, ProbingError)


def test_spec_errors_did_not_gain_value_error():
    # PositionSyntaxError is raised inside pydantic validators; were it a
    # ValueError, pydantic would swallow and re-wrap it.
    from undercurrent.spec import PositionSyntaxError, SpecValidationError

    assert not issubclass(SpecValidationError, ValueError)
    assert not issubclass(PositionSyntaxError, ValueError)


def test_key_error_message_is_not_quoted():
    assert str(ProbingKeyError("no extraction point named 'x'")) == "no extraction point named 'x'"


# ---------------------------------------------------------------------------
# The common mistakes
# ---------------------------------------------------------------------------


@pytest.fixture
def registered_probe():
    from undercurrent.core import Probe, ProbeResult
    from undercurrent.core.registry import default_registry

    class ErrTestProbe(Probe):
        probe_kind = "single_shot"

        def on_start(self, ctx):
            pass

        def on_activation(self, record):
            return None

        def on_end(self, ctx):
            return ProbeResult(ctx.request_id, self.extraction_point_name, verdict=None)

    class ErrTestTrajectory(ErrTestProbe):
        probe_kind = "trajectory"

    default_registry.register("err_test_probe", ErrTestProbe, override=True)
    default_registry.register("err_test_trajectory", ErrTestTrajectory, override=True)
    yield ErrTestProbe
    default_registry.unregister("err_test_probe")
    default_registry.unregister("err_test_trajectory")


def make_record(request_id: str, point_name: str):
    from undercurrent.spec import ActivationRecord

    return ActivationRecord(
        request_id=request_id,
        extraction_point_name=point_name,
        layer=1,
        token_pos=0,
        tensor_type="residual_stream",
        tensor=[0.0],
        is_generated=False,
    )


def raises_probing_error(fn, *substrings: str) -> ProbingError:
    """Run ``fn``; it must raise a ProbingError whose message has every substring."""
    with pytest.raises(ProbingError) as info:
        fn()
    message = str(info.value)
    for substring in substrings:
        assert substring in message, f"{substring!r} not in {message!r}"
    return info.value


def parse(text: str):
    from undercurrent.spec import parse_yaml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return parse_yaml(text)


def test_yaml_key_typo_suggests_the_key_and_gives_the_line():
    text = spec_with(tensor_type="tensor_typ: residual_stream")
    exc = raises_probing_error(
        lambda: parse(text),
        "line 4: extraction_points[0] 'toxicity'",
        "unknown key 'tensor_typ'",
        "Did you mean 'tensor_type'?",
        "missing required key 'tensor_type'",
    )
    assert len(exc.issues) == 2


def test_top_level_key_typo():
    raises_probing_error(
        lambda: parse("extraction_point:\n" + GOOD_POINT),
        "unknown key 'extraction_point'",
        "Did you mean 'extraction_points'?",
    )


def test_bad_tensor_type_suggests_the_value():
    raises_probing_error(
        lambda: parse(spec_with(tensor_type="tensor_type: residual")),
        "line 4: extraction_points[0] 'toxicity'.tensor_type",
        "'residual' is not a valid tensor_type",
        "Did you mean 'residual_stream'?",
        "'attn_out'",
    )


@pytest.mark.parametrize(
    ("line", "field", "suggestion"),
    [
        ("probe_kind: trajectroy", "probe_kind", "'trajectory'"),
        ("execution_mode: asynch", "execution_mode", "'async'"),
    ],
)
def test_bad_enum_values_suggest_the_value(line, field, suggestion):
    overrides = {field: line} if field == "probe_kind" else {"extra": line}
    raises_probing_error(lambda: parse(spec_with(**overrides)), f"not a valid {field}", f"Did you mean {suggestion}?")


def test_bad_position_syntax_names_the_point_and_line():
    raises_probing_error(
        lambda: parse(spec_with(position='position: "generated[x]"')),
        "line 5: extraction_points[0] 'toxicity'.position",
        "invalid position 'generated[x]'",
        "'generated[*]'",
    )


def test_spec_file_errors_are_prefixed_with_path_and_line(tmp_path):
    from undercurrent.spec import load_yaml_file

    path = tmp_path / "probes.yaml"
    path.write_text(spec_with(layers="layers: [-1]"), encoding="utf-8")
    raises_probing_error(
        lambda: load_yaml_file(path),
        f"{path}:3: extraction_points[0] 'toxicity'.layers",
        "must be >= 0 (got -1)",
        "undercurrent inspect-model",
    )


def test_invalid_yaml_syntax_gives_the_line():
    raises_probing_error(lambda: parse("extraction_points: [\n  - x"), "invalid YAML", "line 2", "indentation")


def test_duplicate_names_in_spec():
    text = "extraction_points:\n" + GOOD_POINT + GOOD_POINT
    raises_probing_error(
        lambda: parse(text),
        "duplicate extraction point name: 'toxicity'",
        "extraction_points[0] and extraction_points[1]",
        "rename one",
    )


def test_duplicate_names_in_a_point_list(registered_probe):
    from undercurrent.model import ProbedModel

    point = parse(spec_with()).extraction_points[0]
    raises_probing_error(
        lambda: ProbedModel(None, spec=[point, point]),
        "duplicate extraction point name(s): 'toxicity'",
        "rename",
    )


def test_async_single_shot_combination():
    raises_probing_error(
        lambda: parse(spec_with(extra="execution_mode: async")),
        "extraction_points[0] 'toxicity'",
        "execution_mode='async' is not valid with probe_kind='single_shot'",
        "probe_kind='trajectory'",
    )


def test_unknown_probe_type_suggests_a_registered_one(registered_probe):
    from undercurrent.model import ProbedModel

    point = parse(spec_with(probe_type="probe_type: err_test_prob")).extraction_points[0]
    raises_probing_error(
        lambda: ProbedModel(None, spec=[point]),
        "extraction point 'toxicity'",
        "no probe registered for probe_type='err_test_prob'",
        "Did you mean: 'err_test_probe'",
        "@register_probe(",
    )


def test_probe_kind_mismatch_says_how_to_fix(registered_probe):
    from undercurrent.router import Router

    point = parse(spec_with(probe_type="probe_type: err_test_trajectory")).extraction_points[0]
    router = Router()
    try:
        raises_probing_error(
            lambda: router.register_request("r1", [point], None),
            "extraction point 'toxicity'",
            "implements probe_kind='trajectory'",
            "declares probe_kind='single_shot'",
            "Set probe_kind: trajectory",
        )
    finally:
        router.shutdown()


def test_route_for_an_unregistered_request():
    from undercurrent.router import Router

    record = make_record("never-registered", "toxicity")
    router = Router()
    try:
        raises_probing_error(
            lambda: router.route(record),
            "request_id='never-registered' is not registered",
            "register_request(",
            "router.request(",
        )
    finally:
        router.shutdown()


def test_route_for_an_unknown_extraction_point_suggests_one(registered_probe):
    from undercurrent.core import RequestContext
    from undercurrent.router import Router

    point = parse(spec_with()).extraction_points[0]
    router = Router()
    try:
        router.register_request(
            "r1", [point], RequestContext(request_id="r1", prompt_metadata={}, extraction_point_config=None)
        )
        record = make_record("r1", "toxcity")
        raises_probing_error(lambda: router.route(record), "'toxcity'", "Did you mean 'toxicity'?")
        router.end_request("r1")
    finally:
        router.shutdown()


@pytest.fixture
def tiny_model():
    pytest.importorskip("torch")
    from tests.adapters.hf._helpers import make_tiny_gpt2, make_tiny_tokenizer

    return make_tiny_gpt2(), make_tiny_tokenizer()


def test_layer_out_of_range_in_probed_model(tiny_model, registered_probe):
    from undercurrent.model import ProbedModel

    model, tokenizer = tiny_model
    raises_probing_error(
        lambda: ProbedModel(model, tokenizer=tokenizer, spec=spec_with(layers="layers: [40]")),
        "extraction point 'toxicity': layer 40 is out of range",
        "(2 layers, valid 0..1)",
        "undercurrent inspect-model",
    )


def test_layer_out_of_range_names_the_model_path(tmp_path, tiny_model, registered_probe):
    from undercurrent.model import ProbedModel

    model, tokenizer = tiny_model
    model.save_pretrained(tmp_path)
    raises_probing_error(
        lambda: ProbedModel(str(tmp_path), tokenizer=tokenizer, spec=spec_with(layers="layers: [40]")),
        f"out of range for {tmp_path} (2 layers, valid 0..1)",
        f"`undercurrent inspect-model {tmp_path}`",
    )


def test_unsupported_tensor_type_on_backend(registered_probe):
    from undercurrent.model import ProbedModel

    raises_probing_error(
        lambda: ProbedModel(None, spec=spec_with(tensor_type="tensor_type: kv")),
        "extraction point 'toxicity': tensor_type='kv' is not supported by the 'hf' backend",
        "Use one of:",
        "residual_stream",
    )


def test_unsupported_tensor_type_on_hf_adapter():
    from undercurrent.adapters.hf import HFEngineAdapter

    point = parse(spec_with(tensor_type="tensor_type: kv")).extraction_points[0]
    exc = raises_probing_error(
        lambda: HFEngineAdapter().register_extraction("r1", [point]),
        "extraction point 'toxicity': tensor_type='kv' is not supported by the HF adapter",
        "Use one of:",
    )
    assert "docstring" not in str(exc) and "adapter.py" not in str(exc)


def test_unknown_backend_suggests_one():
    from undercurrent.model import ProbedModel

    exc = raises_probing_error(
        lambda: ProbedModel(None, backend="vlm"), "unknown backend 'vlm'", "Did you mean 'vllm'?"
    )
    assert isinstance(exc, ValueError)


def test_missing_vllm(monkeypatch):
    from undercurrent.adapters import _optional
    from undercurrent.model import ProbedModel

    real_import = _optional._import

    def fake_import(name):
        if name == "vllm":
            raise ModuleNotFoundError("No module named 'vllm'", name="vllm")
        return real_import(name)

    monkeypatch.setattr(_optional, "_import", fake_import)
    exc = raises_probing_error(
        lambda: ProbedModel(None, backend="vllm"),
        "backend='vllm' needs vLLM",
        "existing vLLM environment or image",
        'pip install "undercurrent[vllm]"',
        "docs/compatibility.md",
    )
    assert isinstance(exc, ImportError)


def test_probe_decorator_wrong_signature():
    from undercurrent.core.function_probe import probe

    def two_args(record, threshold_value):
        return 0.0

    exc = raises_probing_error(
        lambda: probe(register=False)(two_args),
        "must take exactly one positional parameter",
        "def two_args(record, *, option=...)",
    )
    assert isinstance(exc, ProbeDefinitionError) and isinstance(exc, TypeError)


def test_probe_decorator_on_a_class():
    from undercurrent.core.function_probe import probe

    class NotAFunction:
        pass

    raises_probing_error(lambda: probe(register=False)(NotAFunction), "@probe decorates a function", "@register_probe(")


def test_function_probe_returning_unsupported_type():
    from undercurrent.core.function_probe import probe

    @probe(register=False)
    def returns_text(record):
        return "toxic"

    instance = returns_text.spawn("r1", "toxicity")
    record = make_record("r1", "toxicity")
    exc = raises_probing_error(
        lambda: instance.on_activation(record),
        "returned str",
        "float, int, bool, ProbeSignal or None",
        "float(...)",
    )
    assert isinstance(exc, ProbingTypeError) and isinstance(exc, TypeError)


def test_probe_class_with_bad_probe_kind():
    from undercurrent.core import Probe

    def define():
        class Bad(Probe):
            probe_kind = "single-shot"

    raises_probing_error(define, "probe_kind must be one of", "Did you mean 'single_shot'?")


def test_cli_missing_spec_file(tmp_path, capsys):
    from undercurrent.cli import main

    missing = tmp_path / "nope.yaml"
    assert main(["validate", str(missing)]) == 1
    out = capsys.readouterr().out
    assert f"{missing}: ERROR 1 issue" in out
    assert "file not found; check the path" in out


def test_load_missing_spec_file(tmp_path):
    from undercurrent.model import ProbedModel

    missing = tmp_path / "probes.yaml"
    exc = raises_probing_error(lambda: ProbedModel(None, spec=str(missing)), f"spec file {str(missing)!r} not found")
    assert isinstance(exc, SpecFileNotFoundError) and isinstance(exc, FileNotFoundError)


def test_cli_unknown_backend_suggests_one(capsys):
    from undercurrent.cli import main

    assert main(["inspect-model", "gpt2", "--backend", "vlm"]) == 2
    assert "unknown backend 'vlm'. Did you mean 'vllm'?" in capsys.readouterr().err


def test_cli_prints_probing_errors_without_traceback(tmp_path, capsys, monkeypatch):
    from undercurrent.cli import inspect_model, main

    def boom(*args, **kwargs):
        raise ProbingRuntimeError("something the user can fix")

    monkeypatch.setattr(inspect_model, "inspect_model", boom)
    assert main(["inspect-model", "gpt2"]) == 1
    err = capsys.readouterr().err
    assert err == "error: something the user can fix\n"


def test_unknown_extraction_point_lookup_suggests_one():
    spec = parse(spec_with())
    exc = raises_probing_error(lambda: spec.get("toxicty"), "no extraction point named 'toxicty'", "'toxicity'")
    assert isinstance(exc, KeyError)


def test_registering_a_probe_name_twice(registered_probe):
    from undercurrent.core import Probe, ProbeResult
    from undercurrent.core.registry import register_probe

    class Other(Probe):
        probe_kind = "single_shot"

        def on_start(self, ctx):
            pass

        def on_activation(self, record):
            return None

        def on_end(self, ctx):
            return ProbeResult(ctx.request_id, self.extraction_point_name, verdict=None)

    exc = raises_probing_error(lambda: register_probe("err_test_probe", Other), "already registered", "override=True")
    assert isinstance(exc, ProbingValueError)

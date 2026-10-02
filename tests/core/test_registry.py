import logging
import threading
from importlib.metadata import EntryPoint

import pytest

from undercurrent.core import Probe, ProbeFactory, ProbeResult, registry
from undercurrent.core.registry import (
    ENTRY_POINT_GROUP,
    ProbeNotFoundError,
    ProbeRegistry,
    get_probe_factory,
    list_probes,
    register_probe,
    unregister_probe,
)


class ThresholdProbe(Probe):
    probe_kind = "single_shot"

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        return None

    def on_end(self, request_ctx):
        return ProbeResult(request_id=self.request_id, extraction_point_name=self.extraction_point_name)


class PluginProbe(ThresholdProbe):
    probe_kind = "trajectory"


class AbstractProbe(Probe):
    probe_kind = "single_shot"


class NoKindProbe(Probe):
    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        return None

    def on_end(self, request_ctx):
        return None


def _fake_entry_points(monkeypatch, *specs):
    eps = [EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP) for name, value in specs]
    calls = []

    def fake(*, group):
        calls.append(group)
        return [ep for ep in eps if ep.group == group]

    monkeypatch.setattr(registry, "entry_points", fake)
    return calls


@pytest.fixture
def reg():
    return ProbeRegistry(load_entry_points=False)


# -- registration -------------------------------------------------------------


def test_decorator_registers_and_returns_class(reg):
    @reg.register("linear_probe")
    class LinearProbe(ThresholdProbe):
        pass

    assert LinearProbe.__name__ == "LinearProbe"
    factory = reg.get("linear_probe")
    assert factory == ProbeFactory(LinearProbe, {})
    probe = factory.spawn("req-1", "ep-1")
    assert isinstance(probe, LinearProbe)


def test_decorator_with_default_kwargs(reg):
    @reg.register("strict", threshold=0.9)
    class StrictProbe(ThresholdProbe):
        pass

    assert reg.get("strict").spawn("req-1", "ep-1").threshold == 0.9


def test_functional_form_with_default_kwargs(reg):
    factory = reg.register("strict_linear", ThresholdProbe, threshold=0.9)
    assert factory == ProbeFactory(ThresholdProbe, {"threshold": 0.9})
    assert reg.get("strict_linear") is factory
    assert reg.get("strict_linear").spawn("req-1", "ep-1").threshold == 0.9


def test_functional_form_accepts_a_probe_factory(reg):
    reg.register("from_factory", ProbeFactory(ThresholdProbe, {"threshold": 0.1}), threshold=0.2)
    assert reg.get("from_factory").probe_kwargs == {"threshold": 0.2}


def test_duplicate_name_raises(reg):
    reg.register("dup", ThresholdProbe)

    class Other(ThresholdProbe):
        pass

    with pytest.raises(ValueError, match=r"already registered.*override=True"):
        reg.register("dup", Other)
    assert reg.get("dup").probe_cls is ThresholdProbe


def test_override_replaces(reg):
    reg.register("dup", ThresholdProbe)

    class Other(ThresholdProbe):
        pass

    reg.register("dup", Other, override=True)
    assert reg.get("dup").probe_cls is Other


def test_redefining_the_same_class_replaces_without_override(reg):
    # A re-run notebook cell defines a new class object with the same
    # module and qualname; that must not raise.
    def define():
        @reg.register("cell_probe")
        class CellProbe(ThresholdProbe):
            pass

        return CellProbe

    define()
    second = define()
    assert reg.get("cell_probe").probe_cls is second


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (dict, "expected a subclass of undercurrent.core.Probe"),
        (ThresholdProbe(), "expected a subclass"),
        (NoKindProbe, "probe_kind must be one of"),
        (AbstractProbe, "is abstract"),
    ],
)
def test_invalid_probe_rejected(reg, bad, match):
    with pytest.raises(TypeError, match=match):
        reg.register("bad", bad)
    assert "bad" not in reg


@pytest.mark.parametrize("name", ["", None, 3])
def test_invalid_name_rejected(reg, name):
    with pytest.raises(ValueError, match="non-empty string"):
        reg.register(name, ThresholdProbe)


def test_unregister(reg):
    reg.register("gone", ThresholdProbe)
    reg.unregister("gone")
    assert "gone" not in reg
    with pytest.raises(ProbeNotFoundError):
        reg.unregister("gone")


# -- lookup -------------------------------------------------------------------


def test_lookup_miss_message(reg):
    reg.register("linear_probe", ThresholdProbe)
    reg.register("mlp_probe", ThresholdProbe)

    with pytest.raises(ProbeNotFoundError) as excinfo:
        reg.get("linear_prob")

    err = excinfo.value
    assert isinstance(err, KeyError)
    msg = str(err)
    assert not msg.startswith(("'", '"'))
    assert "no probe registered for probe_type='linear_prob'" in msg
    assert "Did you mean: 'linear_probe'" in msg
    assert "Known probe types: 'linear_probe', 'mlp_probe'" in msg
    assert "@register_probe('linear_prob')" in msg
    assert ENTRY_POINT_GROUP in msg


def test_lookup_miss_on_empty_registry(reg):
    with pytest.raises(ProbeNotFoundError, match=r"Known probe types: \(none\)"):
        reg.get("anything")


def test_isolated_registries_do_not_share_state():
    a = ProbeRegistry(load_entry_points=False)
    b = ProbeRegistry(load_entry_points=False)
    a.register("only_in_a", ThresholdProbe)

    assert a.list() == ["only_in_a"]
    assert b.list() == []
    assert "only_in_a" not in b
    assert "only_in_a" not in list_probes()


def test_concurrent_registration_is_thread_safe(reg):
    barrier = threading.Barrier(16)
    errors = []

    def worker(i):
        barrier.wait()
        try:
            for j in range(50):
                reg.register(f"p{i}_{j}", ThresholdProbe)
                reg.get(f"p{i}_{j}")
        except Exception as exc:  # pragma: no cover - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(reg) == 16 * 50


# -- module-level functions / default registry ----------------------------------


def test_module_level_functions_use_default_registry():
    @register_probe("_test_registry_default_fn", threshold=0.3)
    class DefaultFnProbe(ThresholdProbe):
        pass

    try:
        assert get_probe_factory("_test_registry_default_fn") == ProbeFactory(DefaultFnProbe, {"threshold": 0.3})
        assert "_test_registry_default_fn" in list_probes()
        assert "_test_registry_default_fn" in registry.default_registry
    finally:
        unregister_probe("_test_registry_default_fn")
    assert "_test_registry_default_fn" not in registry.default_registry


# -- entry-point plugins ----------------------------------------------------------


def test_entry_point_loaded_lazily_on_miss(monkeypatch):
    calls = _fake_entry_points(monkeypatch, ("plugin_probe", f"{__name__}:PluginProbe"))
    reg = ProbeRegistry()
    reg.register("explicit", ThresholdProbe)

    assert reg.get("explicit").probe_cls is ThresholdProbe
    assert calls == []  # a hit never touches plugin metadata
    assert "plugin_probe" not in reg

    assert reg.get("plugin_probe").probe_cls is PluginProbe
    assert calls == [ENTRY_POINT_GROUP]
    assert "plugin_probe" in reg


def test_list_loads_all_entry_points(monkeypatch):
    _fake_entry_points(monkeypatch, ("plugin_a", f"{__name__}:PluginProbe"), ("plugin_b", f"{__name__}:ThresholdProbe"))
    reg = ProbeRegistry()
    assert reg.list() == ["plugin_a", "plugin_b"]


def test_broken_plugins_are_logged_and_skipped(monkeypatch, caplog):
    _fake_entry_points(
        monkeypatch,
        ("missing_module", "undercurrent_no_such_module_xyz:Probe"),
        ("not_a_probe", "builtins:dict"),
        ("good", f"{__name__}:PluginProbe"),
    )
    reg = ProbeRegistry()

    with caplog.at_level(logging.WARNING, logger=registry.__name__):
        assert reg.list() == ["good"]
    assert "missing_module" in caplog.text
    assert "not_a_probe" in caplog.text

    with pytest.raises(ProbeNotFoundError) as excinfo:
        reg.get("not_a_probe")
    # A broken plugin isn't suggested as a known name.
    assert "Known probe types: 'good'." in str(excinfo.value)


def test_broken_plugin_loaded_only_once(monkeypatch):
    _fake_entry_points(monkeypatch, ("broken", "undercurrent_no_such_module_xyz:Probe"))
    loads = []
    real_load = EntryPoint.load
    monkeypatch.setattr(EntryPoint, "load", lambda self: loads.append(self.name) or real_load(self))
    reg = ProbeRegistry()

    for _ in range(3):
        with pytest.raises(ProbeNotFoundError):
            reg.get("broken")
    assert loads == ["broken"]


def test_explicit_registration_wins_over_entry_point(monkeypatch):
    _fake_entry_points(monkeypatch, ("shared_name", f"{__name__}:PluginProbe"))
    reg = ProbeRegistry()
    reg.register("shared_name", ThresholdProbe)

    assert reg.list() == ["shared_name"]
    assert reg.get("shared_name").probe_cls is ThresholdProbe


def test_explicit_registration_replaces_loaded_entry_point_without_override(monkeypatch):
    _fake_entry_points(monkeypatch, ("shared_name", f"{__name__}:PluginProbe"))
    reg = ProbeRegistry()
    assert reg.get("shared_name").probe_cls is PluginProbe

    reg.register("shared_name", ThresholdProbe)
    assert reg.get("shared_name").probe_cls is ThresholdProbe


def test_miss_message_suggests_unloaded_plugins(monkeypatch):
    _fake_entry_points(monkeypatch, ("toxicity_probe", f"{__name__}:PluginProbe"))
    reg = ProbeRegistry()
    with pytest.raises(ProbeNotFoundError, match="Did you mean: 'toxicity_probe'"):
        reg.get("toxicity_prob")


def test_entry_points_disabled(monkeypatch):
    calls = _fake_entry_points(monkeypatch, ("plugin_probe", f"{__name__}:PluginProbe"))
    reg = ProbeRegistry(load_entry_points=False)
    with pytest.raises(ProbeNotFoundError):
        reg.get("plugin_probe")
    assert reg.list() == []
    assert calls == []


def test_unreadable_entry_point_metadata_is_skipped(monkeypatch, caplog):
    def boom(*, group):
        raise RuntimeError("corrupt metadata")

    monkeypatch.setattr(registry, "entry_points", boom)
    reg = ProbeRegistry()
    with caplog.at_level(logging.WARNING, logger=registry.__name__):
        assert reg.list() == []
    assert "could not read" in caplog.text

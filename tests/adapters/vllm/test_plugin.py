"""Tests for `undercurrent.adapters.vllm.plugin` (the `vllm.general_plugins` entry
point). No torch or vLLM needed -- `register()` is pure stdlib (logging +
`os.environ`), by design (see plugin.py's module docstring: it must be
importable and callable with zero arguments in every process a `vllm serve`
invocation starts, including ones that never construct an engine)."""

import importlib

import pytest

from undercurrent.adapters.vllm import plugin


@pytest.fixture(autouse=True)
def _reset_registered_flag(monkeypatch):
    """`register()` is idempotent per-process by design (see its module
    docstring) -- reset that guard between tests so each test observes a
    "fresh process" call."""
    monkeypatch.setattr(plugin, "_REGISTERED", False)
    yield


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.delenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", raising=False)
    yield


def test_register_is_importable_and_callable():
    assert callable(plugin.register)
    plugin.register()  # must not raise with zero args, per the entry-point contract


def test_register_sets_env_defaults_if_unset():
    plugin.register()
    assert __import__("os").environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert __import__("os").environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] == "1"


def test_register_does_not_override_explicit_env(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    plugin.register()
    assert __import__("os").environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_register_is_idempotent_across_repeated_calls():
    plugin.register()
    plugin.register()  # vLLM's loader may call this more than once per process; must not raise


def test_register_does_not_import_vllm():
    """Guard against regressing the deliberate scoping decision: this plugin
    must not attempt to import/construct/mutate any vLLM config object --
    that's confirmed unsupported from a zero-arg plugin hook (see module
    docstring). Proxy check: register() must succeed without importing
    `vllm` at all (it must work purely off stdlib), regardless of whether
    a real vLLM install is present in this test environment."""
    import sys

    before = set(sys.modules)
    plugin.register()
    after = set(sys.modules)
    assert "vllm" not in (after - before)


def test_process_role_returns_a_string():
    role = plugin._process_role()
    assert isinstance(role, str)
    assert role  # non-empty


def test_module_reimport_is_safe():
    importlib.reload(plugin)


def test_vllm_general_plugins_entry_point_resolves_to_register():
    """The root pyproject declares `undercurrent.adapters.vllm.plugin:register`
    under `vllm.general_plugins`; vLLM only logs (never raises) when a plugin
    fails to load, so a dangling target would go unnoticed without this."""
    from importlib.metadata import entry_points

    eps = [ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "undercurrent"]
    if not eps:
        pytest.skip("undercurrent is not installed with its entry-point metadata (pip install -e .)")
    (ep,) = eps
    assert ep.value == "undercurrent.adapters.vllm.plugin:register"
    loaded = ep.load()
    # Compare by name, not identity: test_module_reimport_is_safe reloads the module.
    assert (loaded.__module__, loaded.__qualname__) == ("undercurrent.adapters.vllm.plugin", "register")

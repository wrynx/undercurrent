"""Probe registry: look probes up by their ``probe_type`` name.

A spec names the probe that runs at each extraction point with a plain
string, ``probe_type``. The registry maps those strings to
`ProbeFactory` values, so a probe is
registered once and then referred to by name from specs, the Router and
the CLI::

    from undercurrent.core import Probe
    from undercurrent.core.registry import register_probe

    @register_probe("linear_probe")          # decorator form
    class LinearProbe(Probe):
        probe_kind = "single_shot"
        ...

    register_probe("strict_linear", LinearProbe, threshold=0.9)  # functional form, default kwargs

The module-level functions (`register_probe()`, `get_probe_factory()`,
`list_probes()`, `unregister_probe()`) operate on
`default_registry`. Tests and applications that want isolation can
hold their own `ProbeRegistry` instead.

Plugins
-------
Third-party packages can make probes available without being imported
explicitly, by declaring an entry point in the ``undercurrent.probes`` group
(name = ``probe_type``, value = ``module:ProbeClass``)::

    # pyproject.toml of a third-party package
    [project.entry-points."undercurrent.probes"]
    my_probe = "my_package.probes:MyProbe"

Entry points are loaded lazily: a name is loaded the first time a lookup for
it misses, and `list_probes()` loads them all. A plugin that fails to
import, or doesn't point at a valid ``Probe`` subclass, is logged and skipped.
Explicit registrations always win over entry points.
"""

from __future__ import annotations

import builtins
import difflib
import inspect
import logging
import threading
from collections.abc import Callable, Iterable
from importlib.metadata import EntryPoint, entry_points
from typing import Any, TypeVar, overload

from ..errors import ProbeDefinitionError, ProbingError, ProbingValueError
from .probe import VALID_PROBE_KINDS, Probe, ProbeFactory

logger = logging.getLogger(__name__)

#: Entry-point group third-party packages use to publish probes.
ENTRY_POINT_GROUP = "undercurrent.probes"

_P = TypeVar("_P", bound=type[Probe])


class ProbeNotFoundError(ProbingError, KeyError):
    """Raised when no probe is registered under a requested ``probe_type``.

    Subclasses ``KeyError`` so mapping-style ``except KeyError`` handlers
    keep working. The message lists the known names, the closest matches
    and how to register a probe.
    """

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        self.known = known
        lines = [f"no probe registered for probe_type={name!r}."]
        close = difflib.get_close_matches(name, known, n=3)
        if close:
            lines.append(f"Did you mean: {', '.join(repr(c) for c in close)}?")
        lines.append(f"Known probe types: {', '.join(repr(k) for k in known) if known else '(none)'}.")
        lines.append(
            f"Register one with @register_probe({name!r}) on a Probe subclass, or publish it from a "
            f"package via an entry point in the {ENTRY_POINT_GROUP!r} group."
        )
        self.message = " ".join(lines)
        super().__init__(self.message)

    def __str__(self) -> str:
        # KeyError.__str__ would repr() the message, wrapping it in quotes.
        return self.message


def _validate_probe_cls(name: str, probe_cls: Any) -> None:
    if not (isinstance(probe_cls, type) and issubclass(probe_cls, Probe)):
        raise ProbeDefinitionError(
            f"cannot register probe_type={name!r}: expected a subclass of undercurrent.core.Probe, "
            f"got {probe_cls!r}. Subclass Probe and implement on_start, on_activation and on_end."
        )
    kind = getattr(probe_cls, "probe_kind", None)
    if kind not in VALID_PROBE_KINDS:
        raise ProbeDefinitionError(
            f"cannot register probe_type={name!r}: {probe_cls.__name__}.probe_kind must be one of "
            f"{VALID_PROBE_KINDS}, got {kind!r}. Set a class attribute, e.g. probe_kind = 'single_shot'."
        )
    if inspect.isabstract(probe_cls):
        missing = ", ".join(sorted(getattr(probe_cls, "__abstractmethods__", ())))
        raise ProbeDefinitionError(
            f"cannot register probe_type={name!r}: {probe_cls.__name__} is abstract "
            f"(unimplemented: {missing}). Implement them before registering."
        )


def _as_factory(name: str, probe: type[Probe] | ProbeFactory, probe_kwargs: dict[str, Any]) -> ProbeFactory:
    if isinstance(probe, ProbeFactory):
        _validate_probe_cls(name, probe.probe_cls)
        return ProbeFactory(probe.probe_cls, {**probe.probe_kwargs, **probe_kwargs})
    _validate_probe_cls(name, probe)
    return ProbeFactory(probe, dict(probe_kwargs))


def _is_redefinition(old: type[Probe], new: type[Probe]) -> bool:
    """True when ``new`` looks like the same class defined again (a re-run
    notebook cell, a reloaded module) rather than a different probe."""
    return old.__module__ == new.__module__ and old.__qualname__ == new.__qualname__


class ProbeRegistry:
    """A thread-safe mapping of ``probe_type`` names to [`ProbeFactory`][undercurrent.core.ProbeFactory].

    Args:
        load_entry_points: whether lookup misses and ``list()`` consult
            installed plugins in the ``undercurrent.probes`` entry-point group.
            Defaults to True; pass False for a fully self-contained registry.
    """

    def __init__(self, *, load_entry_points: bool = True) -> None:
        self._factories: dict[str, ProbeFactory] = {}
        self._from_entry_points: set[str] = set()
        self._load_entry_points = load_entry_points
        self._entry_points: dict[str, EntryPoint] | None = None  # discovered lazily
        self._tried_entry_points: set[str] = set()
        self._lock = threading.RLock()

    # -- registration -------------------------------------------------------

    @overload
    def register(self, name: str, /, *, override: bool = ..., **probe_kwargs: Any) -> Callable[[_P], _P]: ...

    @overload
    def register(
        self, name: str, probe: type[Probe] | ProbeFactory, /, *, override: bool = ..., **probe_kwargs: Any
    ) -> ProbeFactory: ...

    def register(
        self,
        name: str,
        probe: type[Probe] | ProbeFactory | None = None,
        /,
        *,
        override: bool = False,
        **probe_kwargs: Any,
    ) -> Any:
        """Register a probe under ``name``.

        Used as a decorator (``@registry.register("name")``) it registers
        the decorated class and returns it unchanged. Called with a ``Probe``
        subclass or a ``ProbeFactory`` it registers that and returns the
        stored factory. Extra keyword arguments become default spawn kwargs.

        Raises ``ValueError`` if ``name`` is already registered (unless
        ``override=True``, or the new class is a redefinition of the same
        class, e.g. a re-run notebook cell), and ``TypeError`` if the probe
        isn't a concrete ``Probe`` subclass with a valid ``probe_kind``.
        """
        if not isinstance(name, str) or not name:
            raise ProbingValueError(
                f"probe_type name must be a non-empty string (got {name!r}), e.g. @register_probe('my_probe')"
            )

        if probe is None:

            def decorator(probe_cls: _P) -> _P:
                self.register(name, probe_cls, override=override, **probe_kwargs)
                return probe_cls

            return decorator

        factory = _as_factory(name, probe, probe_kwargs)
        with self._lock:
            existing = self._factories.get(name)
            if (
                existing is not None
                and not override
                and name not in self._from_entry_points
                and not _is_redefinition(existing.probe_cls, factory.probe_cls)
            ):
                raise ProbingValueError(
                    f"probe_type {name!r} is already registered to {existing.probe_cls.__qualname__}. "
                    f"Pick another name, or pass override=True to replace it."
                )
            self._factories[name] = factory
            self._from_entry_points.discard(name)
            # An explicit registration settles this name: never load a plugin over it.
            self._tried_entry_points.add(name)
        return factory

    def unregister(self, name: str) -> None:
        """Remove ``name``. Raises [`ProbeNotFoundError`][undercurrent.core.ProbeNotFoundError] if absent."""
        with self._lock:
            if name not in self._factories:
                raise ProbeNotFoundError(name, sorted(self._factories))
            del self._factories[name]
            self._from_entry_points.discard(name)

    # -- lookup -------------------------------------------------------------

    def get(self, name: str) -> ProbeFactory:
        """Return the [`ProbeFactory`][undercurrent.core.ProbeFactory] registered under ``name``.

        On a miss, tries a matching ``undercurrent.probes`` entry point
        before raising [`ProbeNotFoundError`][undercurrent.core.ProbeNotFoundError].
        """
        with self._lock:
            factory = self._factories.get(name)
            if factory is not None:
                return factory
            if self._load_entry_points:
                self._load_entry_point(name)
                factory = self._factories.get(name)
                if factory is not None:
                    return factory
            raise ProbeNotFoundError(name, self._known_names())

    def list(self) -> list[str]:
        """Sorted names of every registered probe, including all loadable plugins."""
        with self._lock:
            if self._load_entry_points:
                for name in tuple(self._discover_entry_points()):
                    self._load_entry_point(name)
            return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._factories

    def __len__(self) -> int:
        with self._lock:
            return len(self._factories)

    def __repr__(self) -> str:
        with self._lock:
            return f"{type(self).__name__}({sorted(self._factories)})"

    # -- entry points -------------------------------------------------------

    def _known_names(self) -> builtins.list[str]:  # `list` is shadowed by the method above
        names = set(self._factories)
        if self._load_entry_points:
            # Unloaded but discoverable plugins are worth suggesting too
            # (broken ones have already been tried and are left out).
            names.update(n for n in self._discover_entry_points() if n not in self._tried_entry_points)
        return sorted(names)

    def _discover_entry_points(self) -> dict[str, EntryPoint]:
        if self._entry_points is None:
            found: Iterable[EntryPoint]
            try:
                found = entry_points(group=ENTRY_POINT_GROUP)
            except Exception:  # a corrupt distribution's metadata must not break lookups
                logger.warning("could not read %r entry points; skipping plugins", ENTRY_POINT_GROUP, exc_info=True)
                found = ()
            discovered: dict[str, EntryPoint] = {}
            for ep in found:
                if ep.name in discovered:
                    logger.warning(
                        "probe plugin %r is declared by more than one package (%s, %s); using the first",
                        ep.name,
                        discovered[ep.name].value,
                        ep.value,
                    )
                    continue
                discovered[ep.name] = ep
            self._entry_points = discovered
        return self._entry_points

    def _load_entry_point(self, name: str) -> None:
        """Load the plugin for ``name`` at most once; log and skip failures."""
        if name in self._tried_entry_points:
            return
        ep = self._discover_entry_points().get(name)
        if ep is None:
            return
        self._tried_entry_points.add(name)
        try:
            factory = _as_factory(name, ep.load(), {})
        except Exception:
            logger.warning(
                "skipping probe plugin %r (%s) from the %r entry-point group: it failed to load",
                name,
                ep.value,
                ENTRY_POINT_GROUP,
                exc_info=True,
            )
            return
        if name in self._factories:
            # The plugin module registered it explicitly while importing; keep that.
            return
        self._factories[name] = factory
        self._from_entry_points.add(name)


default_registry = ProbeRegistry()
"""The process-wide registry behind the module-level functions and ``Router()``."""


@overload
def register_probe(name: str, /, *, override: bool = ..., **probe_kwargs: Any) -> Callable[[_P], _P]: ...


@overload
def register_probe(
    name: str, probe: type[Probe] | ProbeFactory, /, *, override: bool = ..., **probe_kwargs: Any
) -> ProbeFactory: ...


def register_probe(
    name: str,
    probe: type[Probe] | ProbeFactory | None = None,
    /,
    *,
    override: bool = False,
    **probe_kwargs: Any,
) -> Any:
    """Register a probe under a ``probe_type`` name in the global registry.

    Specs, ``ProbedModel`` and the ``Router`` then find it by that name.

    ```python
    @register_probe("linear_probe")          # decorator form
    class LinearProbe(Probe):
        probe_kind = "single_shot"
        ...

    register_probe("strict_linear", LinearProbe, threshold=0.9)  # functional form, default kwargs
    ```

    Third-party packages can also make probes available without being
    imported, through an entry point in the ``undercurrent.probes`` group
    (name = ``probe_type``, value = ``module:ProbeClass``):

    ```toml
    [project.entry-points."undercurrent.probes"]
    my_probe = "my_package.probes:MyProbe"
    ```

    Entry points are loaded the first time a lookup for their name misses.
    Explicit registrations always win over entry points.

    Operates on [`default_registry`][undercurrent.core.default_registry];
    see [`ProbeRegistry.register`][undercurrent.core.ProbeRegistry.register]
    for the arguments and errors.
    """
    if probe is None:
        return default_registry.register(name, override=override, **probe_kwargs)
    return default_registry.register(name, probe, override=override, **probe_kwargs)


def unregister_probe(name: str) -> None:
    """Remove ``name`` from [`default_registry`][undercurrent.core.default_registry]."""
    default_registry.unregister(name)


def get_probe_factory(name: str) -> ProbeFactory:
    """Look ``name`` up in [`default_registry`][undercurrent.core.default_registry].

    Raises:
        ProbeNotFoundError: nothing is registered under ``name``.
    """
    return default_registry.get(name)


def list_probes() -> list[str]:
    """Sorted names of every probe in [`default_registry`][undercurrent.core.default_registry], including plugins."""
    return default_registry.list()


__all__ = [
    "ENTRY_POINT_GROUP",
    "ProbeNotFoundError",
    "ProbeRegistry",
    "default_registry",
    "get_probe_factory",
    "list_probes",
    "register_probe",
    "unregister_probe",
]

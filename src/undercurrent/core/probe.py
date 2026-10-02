"""Probe: the abstract base class every concrete probe implements.

Isolation model
----------------
A `Probe` *subclass* is a stateless, reusable factory -- it carries only
behavior, registered once (e.g. in a probe registry keyed by `probe_type`)
and reused across every request that references it. Actual *instances* are
cheap and single-use: one is spawned fresh per (request_id,
extraction_point_name) pair via `Probe.spawn(...)`, used for exactly one
request's lifecycle (`on_start` -> `on_activation`* -> `on_end`), and then
discarded. Instances must never be reused or shared across requests,
extraction points, or concurrent calls.

`Probe.spawn(...)` is the only sanctioned way to construct an instance --
call it on the subclass, never on `Probe` itself, and never call a
subclass's `__init__` directly in probe/router code (tests are the
exception: constructing an unspawned instance to unit-test a single method
in isolation is fine, but `.request_id` / `.extraction_point_name` will
raise until it's been spawned).

To keep this a real guarantee rather than a docstring promise,
`__init_subclass__` rejects any subclass that defines a class-level
list/dict/set/bytearray attribute: mutable class attributes are exactly the
mechanism by which "isolated" instances end up silently sharing state, so
that pattern is refused outright at class-definition time. Put per-request
mutable state on `self`, assigned in `__init__` or `on_start`, instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..errors import ProbeDefinitionError, ProbingRuntimeError, ProbingTypeError, ProbingValueError, did_you_mean
from .activation import ActivationRecord
from .context import RequestContext
from .result import ProbeResult
from .signal import ProbeSignal

VALID_PROBE_KINDS = ("single_shot", "trajectory")
_DISALLOWED_CLASS_ATTR_TYPES = (list, dict, set, bytearray)


class Probe(ABC):
    """Base class for all probes.

    Subclass it for stateful or trajectory probes (for a stateless function,
    [`@probe`][undercurrent.core.function_probe.probe] is shorter). Set
    ``probe_kind`` to ``"single_shot"`` or ``"trajectory"`` and implement the
    three lifecycle methods:

    ```python
    @register_probe("mean_norm")
    class MeanNorm(Probe):
        probe_kind = "trajectory"

        def __init__(self, threshold: float = 10.0):
            super().__init__()
            self.threshold = threshold

        def on_start(self, request_ctx):
            self.norms = []

        def on_activation(self, record):
            self.norms.append(float(record.tensor.norm()))
            return None

        def on_end(self, request_ctx):
            mean = sum(self.norms) / max(len(self.norms), 1)
            return ProbeResult(self.request_id, self.extraction_point_name, verdict=mean > self.threshold)
    ```

    Isolation: the subclass is a stateless, reusable factory. A fresh
    instance is spawned per (request, extraction point) with
    [`spawn`][undercurrent.core.Probe.spawn], used for exactly one
    request (``on_start`` -> ``on_activation``* -> ``on_end``) and then
    discarded. Keep per-request state on ``self``, set in ``__init__`` or
    ``on_start``; a class-level list, dict, set or bytearray attribute is
    rejected when the class is defined, because every instance would share
    it.

    Attributes:
        probe_kind: ``"single_shot"`` (one activation in, one verdict out)
            or ``"trajectory"`` (state carried across many activations).
    """

    probe_kind: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        for name, value in vars(cls).items():
            if name.startswith("__"):
                continue
            if isinstance(value, _DISALLOWED_CLASS_ATTR_TYPES):
                raise ProbeDefinitionError(
                    f"{cls.__name__}.{name} is a class-level {type(value).__name__}, which "
                    "would be shared by every spawned instance. Probe subclasses must stay "
                    "stateless factories -- set per-request mutable state on `self` inside "
                    "__init__ or on_start instead."
                )

        if "probe_kind" in vars(cls) and cls.probe_kind not in VALID_PROBE_KINDS:
            raise ProbeDefinitionError(
                f"{cls.__name__}.probe_kind must be one of {VALID_PROBE_KINDS}, got {cls.probe_kind!r}."
                f"{did_you_mean(cls.probe_kind, VALID_PROBE_KINDS)} Use 'single_shot' for a probe that scores "
                "each activation on its own, 'trajectory' for one that keeps state across a generation."
            )

    def __init__(self) -> None:
        self._request_id: str | None = None
        self._extraction_point_name: str | None = None

    @property
    def request_id(self) -> str:
        """The request this instance was spawned for."""
        if self._request_id is None:
            raise ProbingRuntimeError(
                f"{type(self).__name__} has no request_id -- it must be constructed via "
                f"{type(self).__name__}.spawn(...), not called directly"
            )
        return self._request_id

    @property
    def extraction_point_name(self) -> str:
        """The extraction point this instance was spawned for."""
        if self._extraction_point_name is None:
            raise ProbingRuntimeError(
                f"{type(self).__name__} has no extraction_point_name -- it must be constructed "
                f"via {type(self).__name__}.spawn(...), not called directly"
            )
        return self._extraction_point_name

    @classmethod
    def spawn(cls, request_id: str, extraction_point_name: str, **probe_kwargs: Any) -> Probe:
        """Create a fresh probe instance scoped to one (request_id, extraction_point_name).

        Every call returns a brand-new instance with its own state; nothing
        is shared with any other spawned instance, even of the same
        subclass constructed with identical kwargs. `probe_kwargs` are
        forwarded to the subclass's `__init__` (e.g. a classification
        threshold), and are themselves fresh per call -- pass plain values,
        not shared mutable containers, if you want that guarantee to hold.
        """
        if cls is Probe:
            raise ProbingTypeError("Probe is abstract; call .spawn() on a concrete subclass")
        if getattr(cls, "probe_kind", None) not in VALID_PROBE_KINDS:
            raise ProbeDefinitionError(
                f"{cls.__name__}.probe_kind must be one of {VALID_PROBE_KINDS}, "
                f"got {getattr(cls, 'probe_kind', None)!r}. Set a class attribute, e.g. probe_kind = 'single_shot'."
            )
        if not request_id:
            raise ProbingValueError(f"Probe.spawn() requires a non-empty request_id (got {request_id!r})")
        if not extraction_point_name:
            raise ProbingValueError(
                f"Probe.spawn() requires a non-empty extraction_point_name (got {extraction_point_name!r})"
            )

        instance = cls(**probe_kwargs)
        instance._request_id = request_id
        instance._extraction_point_name = extraction_point_name
        return instance

    @abstractmethod
    def on_start(self, request_ctx: RequestContext) -> None:
        """Called once, before any activations, with the full request context."""

    @abstractmethod
    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        """Called once per matching activation.

        May return a `ProbeSignal` immediately -- useful for inline probes
        that can intervene mid-generation (e.g. `action=abort`) -- or
        `None` if this activation doesn't warrant one.
        """

    @abstractmethod
    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        """Called exactly once, when generation ends or is aborted.

        Must always return a `ProbeResult`, even if `on_activation` was
        never called (e.g. the extraction point never matched a token).
        """


@dataclass(frozen=True)
class ProbeFactory:
    """Binds a Probe subclass to fixed spawn kwargs.

    A single value representing "the thing that produces probes for this
    extraction point", as stored in a probe registry or passed to
    ``Router(probe_registry={...})``.

    Attributes:
        probe_cls: the [`Probe`][undercurrent.core.Probe] subclass to spawn.
        probe_kwargs: keyword arguments passed to its ``__init__``.
    """

    probe_cls: type[Probe]
    probe_kwargs: dict[str, Any] = field(default_factory=dict)

    def spawn(self, request_id: str, extraction_point_name: str, **extra_kwargs: Any) -> Probe:
        """Spawn a probe with ``probe_kwargs`` merged with ``extra_kwargs``
        (``extra_kwargs`` wins on a key clash -- the router passes an
        extraction point's ``probe_args`` here)."""
        return self.probe_cls.spawn(request_id, extraction_point_name, **{**self.probe_kwargs, **extra_kwargs})

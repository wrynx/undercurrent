"""Fully resolved, adapter-facing spec types.

This is the contract other packages (adapters, router, probes) should
import and depend on. Nothing here re-parses YAML or grammar strings --
that work has already been done by `undercurrent.spec.parser`. These types
are plain frozen dataclasses so they carry no framework-specific behavior.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..errors import ProbingKeyError, ProbingValueError, did_you_mean
from .position import PositionSelector
from .schema import ExecutionMode, InterventionMode, ProbeKind, TensorType, TimeoutAction, UntilKind


class FrozenArgs(Mapping[str, Any]):
    """Read-only, picklable mapping used for `ExtractionPoint.probe_args`.

    It copies its input, so mutating the dict it was built from doesn't
    change the extraction point. Compares equal to any mapping with the same
    items (including a plain dict).
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenArgs({self._data!r})"

    def __reduce__(self) -> tuple[type[FrozenArgs], tuple[dict[str, Any]]]:
        return (FrozenArgs, (self._data,))


@dataclass(frozen=True)
class InterventionPolicy:
    """How an inline extraction point's probe may hold up generation.

    The all-defaults instance (``InterventionPolicy()``) is the no-op
    default: ``mode=reject``, ``timeout_ms=None``, ``on_timeout=continue``.

    ```python
    InterventionPolicy(mode="block_until_signal", timeout_ms=50, on_timeout="abort")
    ```

    Attributes:
        mode: see [`InterventionMode`][undercurrent.spec.InterventionMode].
        timeout_ms: how long ``block_until_signal`` waits for the probe, in
            milliseconds. Required (``>= 1``) for that mode, None otherwise.
        on_timeout: what to do when the wait times out or the probe raises;
            see [`TimeoutAction`][undercurrent.spec.TimeoutAction].

    Raises:
        ProbingValueError: ``mode="block_until_signal"`` without a
            ``timeout_ms >= 1``.
    """

    mode: InterventionMode = InterventionMode.REJECT
    timeout_ms: int | None = None
    on_timeout: TimeoutAction = TimeoutAction.CONTINUE

    def __post_init__(self) -> None:
        # Same rule InterventionPolicySpec enforces for YAML specs, for policies
        # built directly in Python: block_until_signal must never wait forever.
        if self.mode == InterventionMode.BLOCK_UNTIL_SIGNAL and (self.timeout_ms is None or self.timeout_ms < 1):
            raise ProbingValueError(
                f"InterventionPolicy: timeout_ms must be an int >= 1 when mode='block_until_signal' "
                f"(got {self.timeout_ms!r}). Pass e.g. InterventionPolicy(mode='block_until_signal', timeout_ms=50)"
            )


@dataclass(frozen=True)
class ExtractionPoint:
    """What to capture, where, and which probe gets it.

    One entry of a spec's ``extraction_points``, fully resolved: everything
    an adapter needs to decide, for each token, whether to capture an
    activation and which probe to hand it to. Usually built from YAML with
    [`load_spec`][undercurrent.spec.load_spec]; build one directly with
    [`parse_position`][undercurrent.spec.parse_position] for ``position``:

    ```python
    ExtractionPoint(
        name="last_prompt_token", layers=(6,), tensor_type=TensorType.RESIDUAL_STREAM,
        position=parse_position("prompt[-1]"), stride=None, until=None,
        probe_type="norm", probe_kind=ProbeKind.SINGLE_SHOT,
        execution_mode=ExecutionMode.INLINE, queue_depth=None,
    )
    ```

    Attributes:
        name: unique name within the spec; results are keyed by it.
        layers: decoder-layer indices to capture at, counted from 0.
        tensor_type: which tensor to capture at each layer.
        position: which token positions to capture.
        stride: for a continuous position (``generated[*]`` or a slice),
            capture only every ``stride``-th position. None captures all.
        until: for a continuous position, when to stop capturing (not
            enforced by the shipped adapters yet; see
            [`UntilKind`][undercurrent.spec.UntilKind]).
        probe_type: the registered name of the probe to run.
        probe_kind: the probe's shape; must match the probe's ``probe_kind``.
        execution_mode: inline (can abort) or async (observe only).
        queue_depth: async only: how many activations may wait for the
            probe. None uses the router's default.
        intervention: inline only: the intervention policy. None uses the
            router's default.
        probe_args: extra keyword arguments for the probe's constructor,
            merged over the registered factory's kwargs (these win). Stored
            as a read-only [`FrozenArgs`][undercurrent.spec.FrozenArgs].
    """

    name: str
    layers: tuple[int, ...]
    tensor_type: TensorType
    position: PositionSelector
    stride: int | None
    until: UntilKind | None
    probe_type: str
    probe_kind: ProbeKind
    execution_mode: ExecutionMode
    queue_depth: int | None
    # None means "not specified here" -- undercurrent.router.Router falls back to
    # its own default_intervention_policy in that case.
    intervention: InterventionPolicy | None = None
    # Any mapping passed in is copied into a read-only FrozenArgs. Excluded
    # from __hash__ (its values may be unhashable, e.g. lists) but included
    # in ==.
    probe_args: Mapping[str, Any] = field(default_factory=FrozenArgs, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.probe_args, FrozenArgs):
            object.__setattr__(self, "probe_args", FrozenArgs(self.probe_args))

    def matches(
        self,
        token_index: int,
        is_generated: bool,
        prompt_len: int,
        generated_index: int | None = None,
        num_generated_total: int | None = None,
    ) -> bool:
        """Whether this extraction point should fire for a given token.

        Combines the position selector match with ``stride`` subsampling
        (which applies only to continuous selectors: 'generated[*]' or a
        slice). See
        [`PositionSelector.matches`][undercurrent.spec.PositionSelector.matches]
        for the meaning of each argument.
        """
        if not self.position.matches(
            token_index,
            is_generated,
            prompt_len,
            generated_index=generated_index,
            num_generated_total=num_generated_total,
        ):
            return False

        if self.stride is not None and self.stride > 1 and self.position.is_continuous:
            if generated_index is None:
                return False
            range_start = self.position.slice_start or 0
            offset_in_range = generated_index - range_start
            if offset_in_range % self.stride != 0:
                return False

        return True


@dataclass(frozen=True)
class ProbeSpec:
    """A fully resolved spec: an ordered collection of extraction points.

    Iterable (yields [`ExtractionPoint`][undercurrent.spec.ExtractionPoint]s)
    and sized. Build one with [`load_spec`][undercurrent.spec.load_spec].

    Attributes:
        version: the spec format version (``"1"``).
        extraction_points: the extraction points, in spec order.
    """

    version: str
    extraction_points: tuple[ExtractionPoint, ...]

    def __iter__(self) -> Iterator[ExtractionPoint]:
        return iter(self.extraction_points)

    def __len__(self) -> int:
        return len(self.extraction_points)

    def get(self, name: str) -> ExtractionPoint:
        """Look up an extraction point by name.

        Raises:
            KeyError: no extraction point has that name.
        """
        for point in self.extraction_points:
            if point.name == name:
                return point
        raise ProbingKeyError(
            f"no extraction point named '{name}'.{did_you_mean(name, self.names)} "
            f"Extraction points in this spec: {', '.join(repr(n) for n in self.names) or '(none)'}."
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The extraction point names, in spec order."""
        return tuple(p.name for p in self.extraction_points)

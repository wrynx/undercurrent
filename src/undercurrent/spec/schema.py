"""Pydantic models for the raw, as-written extraction-point spec.

This module is the "surface syntax" layer: it validates that a parsed YAML
document has the right shape and types, and enforces semantic combination
rules (e.g. ``execution_mode=async`` requires ``probe_kind=trajectory``).

It intentionally does not resolve ``position`` strings into
`PositionSelector` objects -- that happens one
layer up, in `undercurrent.spec.parser`, producing the fully resolved
`ExtractionPoint`. Adapters should depend on
that resolved type, not on these pydantic models.
"""

from __future__ import annotations

import sys
import warnings
from enum import Enum
from types import FrameType
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .position import parse_position


class TensorType(str, Enum):
    """Which tensor an extraction point captures at each of its layers (``tensor_type`` in a spec)."""

    RESIDUAL_STREAM = "residual_stream"
    """The decoder layer's output hidden state (the residual stream after the layer)."""
    ATTN_OUT = "attn_out"
    """The attention block's output."""
    MLP_OUT = "mlp_out"
    """The MLP block's output."""
    KV = "kv"
    """The KV cache. Part of the spec vocabulary, but no shipped adapter captures it."""
    # Captured in undercurrent.adapters.vllm.worker_extension._find_final_norm.
    FINAL_NORM = "final_norm"
    """The model-level final norm's output (e.g. Llama's trailing RMSNorm).

    Distinct from ``RESIDUAL_STREAM`` at the last layer, which is the
    pre-norm residual. A probe trained on ``transformers``'
    ``outputs.hidden_states[-1]`` / ``outputs.last_hidden_state`` was
    trained on this post-norm tensor.
    """


class UntilKind(str, Enum):
    """When a continuous extraction point stops capturing (``until`` in a spec).

    Validated and round-tripped, but not enforced by the shipped adapters
    yet: a continuous position runs until generation ends. Use a bounded
    slice such as ``generated[0:64]`` for a fixed window.
    """

    GENERATION_END = "generation_end"
    """Capture until generation ends."""
    FIXED_COUNT = "fixed_count"
    """Capture a fixed number of positions."""
    STOP_TOKEN = "stop_token"
    """Capture until a stop token is generated."""


class ProbeKind(str, Enum):
    """A probe's shape: one activation in, one verdict out (``SINGLE_SHOT``), or
    state carried across many activations (``TRAJECTORY``)."""

    SINGLE_SHOT = "single_shot"
    """One activation in, one verdict out. Runs inline."""
    TRAJECTORY = "trajectory"
    """State carried across many activations. Runs inline or async."""


class ExecutionMode(str, Enum):
    """``INLINE`` probes run on the generation path and can abort it; ``ASYNC``
    probes run on a worker pool and only observe."""

    INLINE = "inline"
    """Run on the generation path; signals can abort generation."""
    ASYNC = "async"
    """Run on a bounded worker pool; observe only. Needs a trajectory probe."""


class InterventionMode(str, Enum):
    """How an inline extraction point's ``ProbeSignal`` relates to the
    engine's decode step."""

    # REJECT is the only mode valid for execution_mode=async: async's
    # fire-and-forget queueing is the same "engine is not held" shape.
    REJECT = "reject"
    """The default. The router calls ``on_activation`` and returns whatever
    it produces, with no timeout or fallback. The only mode valid for
    ``execution_mode=async``."""
    BLOCK_UNTIL_SIGNAL = "block_until_signal"
    """The router waits up to ``timeout_ms`` for ``on_activation`` and
    applies ``on_timeout`` if it doesn't return in time (or raises).
    Requires ``timeout_ms``, so a probe can never hang generation."""


class TimeoutAction(str, Enum):
    """What ``InterventionPolicy`` falls back to when ``timeout_ms`` elapses
    (or the probe raises) during ``block_until_signal`` dispatch."""

    CONTINUE = "continue"
    """Carry on generating as if the probe had returned no signal."""
    ABORT = "abort"
    """Stop generation."""


class InterventionPolicySpec(BaseModel):
    """As-written intervention policy for one extraction point (or a
    router-level default). See ``InterventionMode``/``TimeoutAction`` for
    what each field means.

    The all-defaults instance (``mode=reject``, ``timeout_ms=None``,
    ``on_timeout=continue``) is the "no-op default" referenced elsewhere in
    this platform (e.g. the rule that ``execution_mode=async`` may only use
    the no-op default) -- the combination checks below guarantee that
    ``mode=reject`` is *only* ever the fully-default instance, so "is this
    the no-op default" reduces to a single check: ``mode == REJECT``.
    """

    model_config = ConfigDict(extra="forbid")

    mode: InterventionMode = InterventionMode.REJECT
    timeout_ms: int | None = None
    on_timeout: TimeoutAction = TimeoutAction.CONTINUE

    @model_validator(mode="after")
    def _check_combinations(self) -> InterventionPolicySpec:
        if self.mode == InterventionMode.BLOCK_UNTIL_SIGNAL:
            if self.timeout_ms is None:
                raise ValueError(
                    "intervention.timeout_ms is required when mode='block_until_signal' -- "
                    "a probe must never be able to hang generation indefinitely. "
                    "Add e.g. timeout_ms: 50"
                )
            if self.timeout_ms < 1:
                raise ValueError(
                    f"intervention.timeout_ms must be >= 1 (got {self.timeout_ms!r}). It is how many "
                    "milliseconds generation waits for the probe, e.g. timeout_ms: 50"
                )
        else:
            if self.timeout_ms is not None:
                raise ValueError(
                    f"intervention.timeout_ms is only valid when mode='block_until_signal' (got "
                    f"mode={self.mode.value!r}). Set mode: block_until_signal, or remove timeout_ms"
                )
            if self.on_timeout != TimeoutAction.CONTINUE:
                raise ValueError(
                    f"intervention.on_timeout is only meaningful when mode='block_until_signal' (got "
                    f"mode={self.mode.value!r}). Set mode: block_until_signal, or remove on_timeout"
                )
        return self


#: Deprecated input key -> canonical key. The old keys are still accepted
#: on input (with a ``DeprecationWarning``); the serializer only ever emits
#: the canonical ones.
DEPRECATED_KEY_ALIASES: dict[str, str] = {"layer": "layers", "tensor": "tensor_type"}


class ExtractionPointSpec(BaseModel):
    """As-written extraction point, validated but not yet resolved."""

    model_config = ConfigDict(extra="forbid")

    name: str
    layers: int | list[int]
    tensor_type: TensorType
    position: int | str
    # 'every_n' is accepted as a synonym for 'stride' -- both mean "only
    # match every Nth position within a continuous selector".
    stride: int | None = Field(default=None, validation_alias=AliasChoices("stride", "every_n"))
    until: UntilKind | None = None
    probe_type: str
    probe_kind: ProbeKind
    execution_mode: ExecutionMode = ExecutionMode.INLINE
    queue_depth: int | None = None
    # None means "not specified here" -- undercurrent.router.Router falls back to
    # its own default_intervention_policy in that case (mirrors how
    # queue_depth=None falls back to the router's default_queue_depth).
    intervention: InterventionPolicySpec | None = None
    # Extra constructor kwargs for this point's probe, merged over the
    # router's ProbeFactory.probe_kwargs at spawn time (the point wins).
    probe_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _rename_deprecated_keys(cls, data: Any) -> Any:
        """Map the deprecated ``layer``/``tensor`` keys onto ``layers``/``tensor_type``.

        Giving both the old and the new key for the same field is an error.
        The DeprecationWarning itself is emitted once per parse call by
        ``ProbeSpecFile._warn_deprecated_keys``, not here.
        """
        if not isinstance(data, dict):
            return data
        old_keys = [old for old in DEPRECATED_KEY_ALIASES if old in data]
        if not old_keys:
            return data
        data = dict(data)
        name = data.get("name", "<unnamed>")
        for old in old_keys:
            new = DEPRECATED_KEY_ALIASES[old]
            if new in data:
                raise ValueError(
                    f"extraction point '{name}': both '{old}' (deprecated) and '{new}' are set; use only '{new}'"
                )
            data[new] = data.pop(old)
        return data

    @field_validator("name", "probe_type")
    @classmethod
    def _non_empty(cls, v: str, info: ValidationInfo) -> str:
        if not v or not v.strip():
            raise ValueError(f"'{info.field_name}' must be a non-empty string (got {v!r})")
        return v

    @field_validator("layers")
    @classmethod
    def _valid_layers(cls, v: int | list[int]) -> int | list[int]:
        layers = v if isinstance(v, list) else [v]
        if not layers:
            raise ValueError("'layers' list must not be empty. List at least one layer index, e.g. layers: [6]")
        for layer in layers:
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise ValueError(f"'layers' entries must be ints (got {layer!r}), e.g. layers: [6, 12]")
            if layer < 0:
                raise ValueError(
                    f"'layers' entries must be >= 0 (got {layer}). Count layers from 0; "
                    "`undercurrent inspect-model MODEL` lists a model's layers"
                )
        return v

    @field_validator("stride")
    @classmethod
    def _valid_stride(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"'stride' must be >= 1 (got {v}); stride: 1 captures every position")
        return v

    @field_validator("queue_depth")
    @classmethod
    def _valid_queue_depth(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"'queue_depth' must be >= 1 (got {v}); it is how many activations may wait")
        return v

    @model_validator(mode="after")
    def _check_execution_mode_combinations(self) -> ExtractionPointSpec:
        # The one rejection the spec explicitly calls out: async execution
        # only makes sense for a probe that tracks state across positions.
        # A single_shot probe run async has no defined semantics, so this
        # is a hard parse-time failure, never a warning.
        if self.execution_mode == ExecutionMode.ASYNC and self.probe_kind == ProbeKind.SINGLE_SHOT:
            raise ValueError(
                f"extraction point '{self.name}': execution_mode='async' is not valid "
                "with probe_kind='single_shot'. Async execution implies a probe that "
                "maintains state across calls; use probe_kind='trajectory', or switch "
                "this extraction point to execution_mode='inline'."
            )

        if self.queue_depth is not None and self.execution_mode != ExecutionMode.ASYNC:
            raise ValueError(
                f"extraction point '{self.name}': 'queue_depth' is only valid when "
                f"execution_mode='async' (got execution_mode='{self.execution_mode.value}'). "
                "Remove queue_depth, or set execution_mode: async"
            )

        # async cannot participate in synchronous intervention: it's already
        # fire-and-forget (route() always returns None for it, immediately),
        # so a block_until_signal policy attached to one has no defined
        # semantics -- the router re-checks this defensively at registration
        # time too (an ExtractionPoint can be hand-built, bypassing this
        # parser entirely), but this is the earliest, loudest place to catch
        # it for anyone going through YAML/parse_dict.
        if (
            self.execution_mode == ExecutionMode.ASYNC
            and self.intervention is not None
            and self.intervention.mode != InterventionMode.REJECT
        ):
            raise ValueError(
                f"extraction point '{self.name}': execution_mode='async' cannot use an "
                f"intervention policy other than the default (mode='reject') -- async cannot "
                f"participate in synchronous intervention (got mode={self.intervention.mode.value!r}). "
                "Remove the intervention block, or set execution_mode: inline"
            )

        return self

    @model_validator(mode="after")
    def _check_position_combinations(self) -> ExtractionPointSpec:
        # position parsing is re-used here (not stored) purely to validate
        # combinations against its *kind*; the resolved PositionSelector
        # itself is built later by undercurrent.spec.parser.
        selector = parse_position(self.position)

        if self.stride is not None and not selector.is_continuous:
            raise ValueError(
                f"extraction point '{self.name}': 'stride' is only meaningful for a "
                f"continuous position selector ('generated[*]' or a slice like "
                f"'generated[5:]'), not for position={self.position!r}. Remove stride, or use a continuous position"
            )

        if self.until is not None and not selector.is_continuous:
            raise ValueError(
                f"extraction point '{self.name}': 'until' is only meaningful for a "
                f"continuous position selector ('generated[*]' or a slice like "
                f"'generated[5:]'), not for position={self.position!r}. Remove until, or use a continuous position"
            )

        return self


class ProbeSpecFile(BaseModel):
    """Top-level container for a parsed extraction-point spec file."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    extraction_points: list[ExtractionPointSpec]

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_keys(cls, data: Any) -> Any:
        # One DeprecationWarning per parse call, naming every point that
        # still uses an old key. The renaming itself happens per point in
        # ExtractionPointSpec._rename_deprecated_keys.
        if not isinstance(data, dict) or not isinstance(data.get("extraction_points"), list):
            return data
        uses = []
        for point in data["extraction_points"]:
            if not isinstance(point, dict):
                continue
            for old, new in DEPRECATED_KEY_ALIASES.items():
                if old in point:
                    uses.append(
                        f"extraction point '{point.get('name', '<unnamed>')}' uses '{old}', rename it to '{new}'"
                    )
        if uses:
            warnings.warn(
                "deprecated spec keys: " + "; ".join(uses),
                DeprecationWarning,
                stacklevel=_caller_stacklevel(),
            )
        return data

    @model_validator(mode="after")
    def _check_unique_names(self) -> ProbeSpecFile:
        seen: dict[str, int] = {}
        for index, point in enumerate(self.extraction_points):
            if point.name in seen:
                raise ValueError(
                    f"duplicate extraction point name: '{point.name}' (extraction_points[{seen[point.name]}] "
                    f"and extraction_points[{index}]). Names must be unique; rename one of them"
                )
            seen[point.name] = index
        return self


def _caller_stacklevel() -> int:
    """``warnings.warn`` stacklevel that attributes the warning to the first
    frame outside this package and pydantic, i.e. the user's parse call.

    Python's default filters only show a DeprecationWarning when it is
    attributed to ``__main__``, so pointing it at pydantic internals would
    hide it from script users entirely.
    """
    level = 1
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if not module.startswith(("undercurrent.spec", "pydantic")):
            return level
        frame = frame.f_back
        level += 1
    return level

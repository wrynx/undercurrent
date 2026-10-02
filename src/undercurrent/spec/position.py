"""Position selector grammar and its resolved, adapter-facing representation.

Users write positions as plain YAML scalars:

    position: 5                  # absolute index into the full sequence
    position: "prompt[-1]"       # last prompt token
    position: "prompt[0]"        # first prompt token
    position: "generated[0]"     # first generated token
    position: "generated[-1]"    # last generated token
    position: "generated[*]"     # every generated token (continuous)
    position: "generated[5:]"    # every generated token from index 5 onward
    position: "generated[2:8]"   # generated tokens in [2, 8)
    position: "prompt[-1]+1"     # offset: one token past the last prompt token

``parse_position`` turns any of these into a single `PositionSelector`
value type. Adapters should only ever call ``.matches(...)`` on that value
-- they must never re-parse the grammar themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from ..errors import ProbingValueError
from .errors import PositionSyntaxError


class PositionKind(str, Enum):
    """The form of a ``position`` selector, as parsed into a `PositionSelector`."""

    ABSOLUTE = "absolute"
    """A bare integer, e.g. ``5``."""
    PROMPT_INDEX = "prompt_index"
    """``prompt[i]``."""
    GENERATED_INDEX = "generated_index"
    """``generated[i]``."""
    GENERATED_WILDCARD = "generated_wildcard"
    """``generated[*]`` (continuous)."""
    GENERATED_SLICE = "generated_slice"
    """``generated[start:]`` or ``generated[start:stop]`` (continuous)."""


_ABS_RE = re.compile(r"^(-?\d+)$")
_PROMPT_RE = re.compile(r"^prompt\[(-?\d+)\](?:([+-]\d+))?$")
_GENERATED_POINT_RE = re.compile(r"^generated\[(-?\d+)\](?:([+-]\d+))?$")
_GENERATED_WILDCARD_RE = re.compile(r"^generated\[\*\](?:([+-]\d+))?$")
_GENERATED_SLICE_RE = re.compile(r"^generated\[(\d+):(\d*)\](?:([+-]\d+))?$")

_GRAMMAR_HELP = (
    "expected one of: an absolute integer index (e.g. 5), "
    "'prompt[i]' / 'prompt[-1]', 'generated[i]' / 'generated[-1]', "
    "'generated[*]', 'generated[start:]' or 'generated[start:stop]', "
    "each optionally followed by an offset modifier such as '+1' or '-2'"
)


@dataclass(frozen=True)
class PositionSelector:
    """Normalized, adapter-facing representation of a ``position`` selector.

    This is the only representation adapters should consume. Build it once
    with [`parse_position`][undercurrent.spec.parse_position] and reuse it
    for every token processed during inference.

    Attributes:
        kind: which grammar form this selector was parsed from.
        index: the raw (possibly negative) index for ABSOLUTE / PROMPT_INDEX
            / GENERATED_INDEX selectors. Unused otherwise.
        slice_start: inclusive start of a GENERATED_SLICE range (always >= 0).
        slice_stop: exclusive stop of a GENERATED_SLICE range, or None for
            an open-ended slice ("generated[5:]").
        offset: an additive offset applied after resolving the base index,
            e.g. the "+1" in "prompt[-1]+1" ("read position b at token b+1").
        raw: the original YAML value this was parsed from, kept for error
            messages and lossless round-trip serialization.
    """

    kind: PositionKind
    index: int | None = None
    slice_start: int | None = None
    slice_stop: int | None = None
    offset: int = 0
    # Kept for error messages and debugging only -- excluded from equality
    # so that a selector re-parsed from its own `to_raw()` output (which
    # may differ cosmetically, e.g. int 5 vs the original str "5") still
    # compares equal to the original.
    raw: int | str = field(default="", compare=False)

    @property
    def is_continuous(self) -> bool:
        """True for selectors that can match more than one token position."""
        return self.kind in (PositionKind.GENERATED_WILDCARD, PositionKind.GENERATED_SLICE)

    def _point_index(self) -> int:
        """``index`` for a single-point selector (ABSOLUTE / PROMPT_INDEX / GENERATED_INDEX).

        `parse_position()` always sets it for those kinds; this guards
        selectors constructed directly without one.
        """
        if self.index is None:
            raise ProbingValueError(f"{self.kind.value} position selector has no index")
        return self.index

    def resolve_absolute(
        self,
        prompt_len: int,
        num_generated_total: int | None = None,
    ) -> int | None:
        """Resolve a single-point selector to an absolute sequence index.

        Returns None for continuous selectors (wildcard/slice), which have
        no single absolute index, or when a negative ``generated[i]``
        selector is asked to resolve before the total generated length is
        known (streaming decode without a final count yet).

        Args:
            prompt_len: number of tokens in the prompt.
            num_generated_total: total number of generated tokens so far
                (or at final length), required only to resolve negative
                ``generated[i]`` indices such as ``generated[-1]``.
        """
        if self.kind == PositionKind.ABSOLUTE:
            base = self._point_index()
        elif self.kind == PositionKind.PROMPT_INDEX:
            index = self._point_index()
            base = index if index >= 0 else prompt_len + index
        elif self.kind == PositionKind.GENERATED_INDEX:
            index = self._point_index()
            if index >= 0:
                gen_idx = index
            else:
                if num_generated_total is None:
                    return None
                gen_idx = num_generated_total + index
            base = prompt_len + gen_idx
        else:
            return None
        return base + self.offset

    def matches(
        self,
        token_index: int,
        is_generated: bool,
        prompt_len: int,
        generated_index: int | None = None,
        num_generated_total: int | None = None,
    ) -> bool:
        """Whether this selector matches a given token.

        Args:
            token_index: absolute 0-based index of the token in the full
                (prompt + generated) sequence.
            is_generated: whether this token was generated (vs. part of the
                prompt).
            prompt_len: number of tokens in the prompt.
            generated_index: 0-based index of this token within the
                generated portion (required to match GENERATED_SLICE /
                GENERATED_WILDCARD selectors; derivable as
                ``token_index - prompt_len`` when ``is_generated``).
            num_generated_total: total generated length so far/final, only
                needed to resolve selectors with negative generated indices
                (e.g. ``generated[-1]``) during streaming decode. If this is
                not supplied and it's needed, the selector will not match
                until it is supplied (so re-check after generation ends for
                those selectors, e.g. via `resolve_absolute()`).
        """
        if self.kind in (
            PositionKind.ABSOLUTE,
            PositionKind.PROMPT_INDEX,
            PositionKind.GENERATED_INDEX,
        ):
            resolved = self.resolve_absolute(prompt_len, num_generated_total)
            if resolved is None:
                return False
            return token_index == resolved

        if self.kind == PositionKind.GENERATED_WILDCARD:
            return is_generated

        if self.kind == PositionKind.GENERATED_SLICE:
            if not is_generated or generated_index is None:
                return False
            start = self.slice_start or 0
            if generated_index < start:
                return False
            if self.slice_stop is not None and generated_index >= self.slice_stop:
                return False
            return True

        raise AssertionError(f"unhandled PositionKind: {self.kind}")  # pragma: no cover

    def to_raw(self) -> int | str:
        """Reconstruct the YAML scalar this selector would parse from.

        Used by the serializer for spec -> internal repr -> spec round trips.
        """
        offset_suffix = f"{self.offset:+d}" if self.offset else ""
        if self.kind == PositionKind.ABSOLUTE:
            if offset_suffix:
                return f"{self.index}{offset_suffix}"
            return self._point_index()
        if self.kind == PositionKind.PROMPT_INDEX:
            return f"prompt[{self.index}]{offset_suffix}"
        if self.kind == PositionKind.GENERATED_INDEX:
            return f"generated[{self.index}]{offset_suffix}"
        if self.kind == PositionKind.GENERATED_WILDCARD:
            return f"generated[*]{offset_suffix}"
        if self.kind == PositionKind.GENERATED_SLICE:
            stop = "" if self.slice_stop is None else str(self.slice_stop)
            return f"generated[{self.slice_start or 0}:{stop}]{offset_suffix}"
        raise AssertionError(f"unhandled PositionKind: {self.kind}")  # pragma: no cover


def parse_position(value: int | str) -> PositionSelector:
    """Parse a raw YAML ``position`` value into a [`PositionSelector`][undercurrent.spec.PositionSelector].

    Accepted forms:

    | Value | Selects |
    |-------|---------|
    | ``5`` | absolute index 5 in the full (prompt + generated) sequence |
    | ``"prompt[0]"``, ``"prompt[-1]"`` | the first / last prompt token |
    | ``"generated[0]"``, ``"generated[-1]"`` | the first / last generated token |
    | ``"generated[*]"`` | every generated token (continuous) |
    | ``"generated[5:]"`` | every generated token from index 5 on (continuous) |
    | ``"generated[2:8]"`` | generated tokens 2 to 7 (continuous) |
    | ``"prompt[-1]+1"`` | any form plus an offset: one token past the last prompt token |

    Raises:
        PositionSyntaxError: ``value`` matches none of the forms; the
            message includes the accepted grammar.
    """
    if isinstance(value, bool):
        raise PositionSyntaxError(
            f"invalid position {value!r}: booleans are not valid positions (YAML reads unquoted "
            f"yes/no/true/false as booleans). {_GRAMMAR_HELP}"
        )

    if isinstance(value, int):
        return PositionSelector(kind=PositionKind.ABSOLUTE, index=value, raw=value)

    if not isinstance(value, str):
        raise PositionSyntaxError(f"invalid position {value!r}: must be an int or str, got {type(value).__name__}")

    text = value.strip()

    m = _GENERATED_WILDCARD_RE.match(text)
    if m:
        offset = int(m.group(1)) if m.group(1) else 0
        return PositionSelector(kind=PositionKind.GENERATED_WILDCARD, offset=offset, raw=value)

    m = _GENERATED_SLICE_RE.match(text)
    if m:
        start = int(m.group(1))
        stop_str = m.group(2)
        stop = int(stop_str) if stop_str else None
        offset = int(m.group(3)) if m.group(3) else 0
        if stop is not None and stop <= start:
            raise PositionSyntaxError(
                f"invalid position {value!r}: slice stop ({stop}) must be greater than start ({start}), "
                f"e.g. 'generated[{start}:{start + 1}]', or 'generated[{start}:]' for no end"
            )
        return PositionSelector(
            kind=PositionKind.GENERATED_SLICE, slice_start=start, slice_stop=stop, offset=offset, raw=value
        )

    m = _PROMPT_RE.match(text)
    if m:
        index = int(m.group(1))
        offset = int(m.group(2)) if m.group(2) else 0
        return PositionSelector(kind=PositionKind.PROMPT_INDEX, index=index, offset=offset, raw=value)

    m = _GENERATED_POINT_RE.match(text)
    if m:
        index = int(m.group(1))
        offset = int(m.group(2)) if m.group(2) else 0
        return PositionSelector(kind=PositionKind.GENERATED_INDEX, index=index, offset=offset, raw=value)

    m = _ABS_RE.match(text)
    if m:
        return PositionSelector(kind=PositionKind.ABSOLUTE, index=int(m.group(1)), raw=value)

    raise PositionSyntaxError(f"invalid position {value!r}: {_GRAMMAR_HELP}")

"""Exception types raised by the undercurrent.spec contract layer.

All parse/validation failures surface as one of these, regardless of whether
the underlying failure came from YAML loading, pydantic type-checking, or a
semantic combination check (e.g. single_shot + async). Callers that only want
to catch "my spec was bad" can catch ``SpecValidationError`` without knowing
or caring that pydantic is used internally.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import ProbingError


class ProbingSpecError(ProbingError):
    """Base class for all errors raised by undercurrent.spec."""


class SpecValidationError(ProbingSpecError):
    """Raised when a spec fails validation.

    Covers both structural problems (wrong types, missing fields) and
    semantic problems (invalid field combinations, e.g.
    ``execution_mode=async`` on a ``probe_kind=single_shot`` extraction
    point). This is a hard failure, never a warning: an invalid spec must
    never silently produce a resolved ``ExtractionPoint``.

    ``issues`` holds one self-contained message per problem (with the file
    and line when known, e.g. ``probes.yaml:14: extraction_points[2]
    'drift'.position: ...``); ``str(exc)`` combines them.
    """

    def __init__(self, message: str, *, issues: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.issues: tuple[str, ...] = tuple(issues) if issues else (message,)


class PositionSyntaxError(ProbingSpecError):
    """Raised when a ``position`` string does not match any supported form."""

"""The base exception for everything Undercurrent raises on purpose.

Catch `ProbingError` to handle any error the library raises because of
how it was called or configured (a bad spec, an unknown probe, a layer that
doesn't exist, a missing vLLM, ...)::

    from undercurrent.errors import ProbingError

    try:
        model = ProbedModel.from_pretrained("gpt2", spec="probes.yaml")
    except ProbingError as exc:
        print(f"configuration problem: {exc}")

Every library exception keeps the built-in base it had before this module
existed (``SpecValidationError`` is still a ``ValueError``,
``ProbeNotFoundError`` still a ``KeyError``, ``MissingDependencyError`` still
an ``ImportError``, ...), so existing ``except`` clauses keep working.

Errors raised by *your* code (a probe's ``on_activation``, a sink callback)
pass through unchanged and are not ``ProbingError``s.

This module imports nothing from the rest of the package, so any module can
import it without creating an import cycle.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable


class ProbingError(Exception):
    """Base class for every exception Undercurrent raises deliberately."""


class ProbingValueError(ProbingError, ValueError):
    """A library call got an invalid value (a ``ValueError``)."""


class ProbingTypeError(ProbingError, TypeError):
    """A library call got an object of the wrong type (a ``TypeError``)."""


class ProbingRuntimeError(ProbingError, RuntimeError):
    """A library object was used in a state that doesn't allow it (a ``RuntimeError``)."""


class ProbingKeyError(ProbingError, KeyError):
    """A name lookup missed (a ``KeyError``), with a readable message."""

    def __str__(self) -> str:
        # KeyError.__str__ would repr() the message, wrapping it in quotes.
        return str(self.args[0]) if self.args else ""


class ProbeDefinitionError(ProbingTypeError):
    """A probe class or ``@probe`` function is defined in a way Undercurrent can't use.

    Raised when the class or function is defined, decorated or registered,
    not when it runs.
    """


class SpecFileNotFoundError(ProbingError, FileNotFoundError):
    """A spec file path doesn't exist (a ``FileNotFoundError``)."""


def did_you_mean(value: object, choices: Iterable[str]) -> str:
    """``" Did you mean 'x'?"`` for the closest of ``choices`` to ``value``, or ``""``.

    The leading space lets callers append it to a sentence unconditionally.
    """
    if not isinstance(value, str):
        return ""
    close = difflib.get_close_matches(value, list(choices), n=1)
    if not close:
        close = difflib.get_close_matches(value.lower(), list(choices), n=1)
    return f" Did you mean {close[0]!r}?" if close else ""


def one_of(choices: Iterable[str]) -> str:
    """``"'a', 'b' or 'c'"``, for listing valid values in a message."""
    quoted = [repr(c) for c in choices]
    if len(quoted) <= 1:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} or {quoted[-1]}"


__all__ = [
    "ProbeDefinitionError",
    "ProbingError",
    "ProbingKeyError",
    "ProbingRuntimeError",
    "ProbingTypeError",
    "ProbingValueError",
    "SpecFileNotFoundError",
]

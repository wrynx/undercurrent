"""Errors the CLI reports as ``error: ...`` (no traceback unless --debug)."""

from __future__ import annotations

from ..errors import ProbingError


class CLIError(ProbingError):
    """A failure to report to the user (exit code 1)."""


class UsageError(CLIError):
    """The command was invoked wrongly (exit code 2)."""

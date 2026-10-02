"""Exceptions raised by undercurrent.router."""

from __future__ import annotations

from ..errors import ProbingError


class RouterError(ProbingError):
    """Raised for invalid Router usage.

    For example: an unknown ``probe_type`` or a duplicate ``request_id`` at
    registration, routing a record for a request that isn't registered (or
    has ended), or a record that doesn't match the extraction point it names.
    """

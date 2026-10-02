"""Exception types for undercurrent.adapters.hf (kept dependency-free so
`introspect.py` and `adapter.py` can both raise them)."""

from __future__ import annotations

from ...errors import ProbingError


class HFAdapterLimitationError(ProbingError):
    """Raised for a documented, known limitation of the HF adapter.

    For example: concurrent ``generate()`` calls, an unsupported
    ``tensor_type``, or a model architecture whose decoder layers can't be
    found. A distinct type, so callers can tell it from a bug.
    """

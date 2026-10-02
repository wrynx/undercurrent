"""Re-export of the runtime activation type, `undercurrent.spec.ActivationRecord`.

Kept so `from undercurrent.core import ActivationRecord` works alongside the
rest of the probe interface.
"""

from __future__ import annotations

from ..spec import ActivationRecord

__all__ = ["ActivationRecord"]

"""Framework-agnostic -> torch.Tensor conversion for ActivationRecord.tensor.

`ActivationRecord.tensor` is deliberately array-like and framework-agnostic
(see `undercurrent.spec.activation_record`): a numpy array, a torch.Tensor, a jax
array, or a plain list/tuple, depending on which adapter produced it. This
is the one place in `content_safety_demo` that normalizes it into a flat
`torch.Tensor` for the probes' classifier/recurrence modules to consume --
going through `numpy.asarray` rather than `torch.as_tensor` directly so that
array-likes which only implement `__array__` (e.g. jax arrays) convert
cleanly too.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def to_flat_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).reshape(-1)
    array = np.asarray(value, dtype=np.float32)
    return torch.from_numpy(array).reshape(-1)

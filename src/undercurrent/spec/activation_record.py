"""The shared runtime data type produced by adapters and consumed by probes.

``ActivationRecord`` is deliberately framework-agnostic: ``tensor`` accepts
whatever array-like object the inference engine's adapter produces (a numpy
array, a torch.Tensor, a jax array, ...). This module does not import numpy
or torch, and never will -- pinning to one framework here would leak an
inference-engine dependency into the contract layer that every adapter is
supposed to depend on, not the other way around.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..errors import ProbingValueError


@dataclass
class ActivationRecord:
    """One captured activation, tied to a single extraction point and token.

    Attributes:
        request_id: identifier of the inference request this activation was
            captured during (opaque; adapters define its format).
        extraction_point_name: the ``name`` of the
            [`ExtractionPoint`][undercurrent.spec.ExtractionPoint] that
            produced this record.
        layer: the specific layer this activation was captured from (a
            single int, even if the extraction point specified multiple
            layers -- one record is emitted per matched layer).
        token_pos: absolute 0-based index of the token this activation
            corresponds to, in the full (prompt + generated) sequence.
        tensor_type: the tensor kind captured, e.g. ``"residual_stream"``
            (a [`TensorType`][undercurrent.spec.TensorType] value, stored as
            a plain str).
        tensor: the captured activation itself. Array-like (with the HF and
            vLLM adapters, a CPU ``torch.Tensor``); its shape depends on the
            tensor type.
        is_generated: whether ``token_pos`` falls in the generated portion
            of the sequence (False for prompt tokens).
        timestamp: unix timestamp (seconds) of when this activation was
            captured. Defaults to the time of construction.
    """

    request_id: str
    extraction_point_name: str
    layer: int
    token_pos: int
    tensor_type: str
    tensor: Any
    is_generated: bool
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ProbingValueError(f"ActivationRecord.request_id must be a non-empty string (got {self.request_id!r})")
        if not self.extraction_point_name:
            raise ProbingValueError(
                f"ActivationRecord.extraction_point_name must be a non-empty string (got {self.extraction_point_name!r})"
            )
        if self.layer < 0:
            raise ProbingValueError(f"ActivationRecord.layer must be >= 0 (got {self.layer!r})")
        if self.token_pos < 0:
            raise ProbingValueError(f"ActivationRecord.token_pos must be >= 0 (got {self.token_pos!r})")
        if self.tensor is None:
            raise ProbingValueError("ActivationRecord.tensor must not be None; pass the captured activation")

    def metadata(self) -> dict[str, Any]:
        """All fields except ``tensor``, as a plain dict.

        Useful for logging, indexing, or any other place that wants to
        handle activation metadata without needing to know how to
        serialize the tensor itself (numpy/torch tensors generally aren't
        directly JSON-serializable).
        """
        return {
            "request_id": self.request_id,
            "extraction_point_name": self.extraction_point_name,
            "layer": self.layer,
            "token_pos": self.token_pos,
            "tensor_type": self.tensor_type,
            "is_generated": self.is_generated,
            "timestamp": self.timestamp,
        }

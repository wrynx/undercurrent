"""undercurrent.adapters.hf: the HuggingFace transformers engine adapter for
the activation-probing platform. It implements the `EngineAdapter` contract
from `undercurrent.adapters.base`.

See docs/_legacy/ for the full architecture writeup. Short version::

    from undercurrent.adapters.hf import HFEngineAdapter

    adapter = HFEngineAdapter()
    adapter.load_model("openai-community/gpt2")
    adapter.register_extraction(request_id, extraction_points)
    text = adapter.generate(request_id, prompt, {"max_new_tokens": 32}, router)
    adapter.unregister_extraction(request_id)

Most users want `undercurrent.ProbedModel` (``backend="hf"``, the default),
which drives this adapter for them.

Public surface:
    - `HFEngineAdapter` -- this package's concrete implementation.
    - `HFAdapterLimitationError` -- raised for a documented, known
      limitation of this reference adapter (no concurrent generate() calls,
      an unsupported tensor_type, an unanticipated model architecture).

Internal (importable from their modules, not covered by the API-stability
promise): `stopping_criteria.ProbingStoppingCriteria` (the abort hook the
adapter installs), `introspect` (layer discovery used by the adapter and
`undercurrent inspect-model`) and `errors`. `EngineAdapter` is still
importable from here for backwards compatibility; its public home is
`undercurrent.adapters`.
"""

from ..base import EngineAdapter as EngineAdapter  # compat alias, not in __all__
from .adapter import HFAdapterLimitationError, HFEngineAdapter

__all__ = [
    "HFAdapterLimitationError",
    "HFEngineAdapter",
]

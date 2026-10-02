"""undercurrent.adapters.vllm: the vLLM engine adapter for Undercurrent.

See the vLLM adapter design doc in docs/_legacy/ for the full architecture writeup
(interception strategy, seq_id -> token_pos mapping, process-topology
limitations). Short version::

    from undercurrent.adapters.vllm import VLLMEngineAdapter

    adapter = VLLMEngineAdapter()
    adapter.load_model("openai-community/gpt2")
    adapter.register_extraction(request_id, extraction_points)
    text = adapter.generate(request_id, prompt, {"max_tokens": 32}, router)
    adapter.unregister_extraction(request_id)

Most users want `undercurrent.ProbedModel` with ``backend="vllm"``, which
drives this adapter for them. Importing this package never imports vLLM; the
adapter imports it (and checks the installed version) when it is used.

Public surface:
    - `VLLMEngineAdapter` -- this package's concrete implementation.
    - `VLLMAdapterLimitationError` -- raised for a documented,
      known limitation (unsupported executor topology, an unreconciled
      vLLM-version API seam) rather than letting one fail silently/opaquely.

Experimental internals (importable from their modules, but *not* covered by
the API-stability promise; they read undocumented vLLM internals and change
whenever vLLM does):
    - `seq_mapper` -- the pure-Python seq_id ->
      (request_id, token_pos) translation core (`SeqIdMapper`,
      `StepBatchMetadata`, `TokenRowMapping`, `SeqMapperError`).
    - `worker_extension` -- the vLLM
      `worker_extension_cls` plugin that does the actual hook installation
      and per-step capture inside the worker process.
    - `introspection`, `plugin`, `support` and `version_check`.

`EngineAdapter` is still importable from here for backwards compatibility;
its public home is `undercurrent.adapters`.
"""

from ..base import EngineAdapter as EngineAdapter  # compat alias, not in __all__
from .adapter import VLLMAdapterLimitationError, VLLMEngineAdapter

__all__ = [
    "VLLMAdapterLimitationError",
    "VLLMEngineAdapter",
]

"""Engine adapters: the bridge between an inference engine and the router.

Public surface:
    - `EngineAdapter` -- the engine-agnostic ABC every adapter
      implements (canonical home: `undercurrent.adapters.base`). Implement
      it to probe an engine Undercurrent doesn't ship an adapter for.
    - `MissingDependencyError` -- raised when an adapter needs a
      package that isn't installed (most often vLLM, which is never
      installed by default).

The concrete adapters live in their own subpackages and are not imported
here, so ``import undercurrent.adapters`` never loads torch, transformers or
vLLM:

    - `undercurrent.adapters.hf`: `HFEngineAdapter` (Hugging Face transformers)
    - `undercurrent.adapters.vllm`: `VLLMEngineAdapter` (vLLM; experimental
      internals, see that package's docstring)

Most users never touch an adapter directly: `undercurrent.ProbedModel`
picks one from ``backend="hf"`` / ``backend="vllm"``.
"""

from ._optional import MissingDependencyError
from .base import EngineAdapter

__all__ = ["EngineAdapter", "MissingDependencyError"]

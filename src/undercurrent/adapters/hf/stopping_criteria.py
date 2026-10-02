"""ProbingStoppingCriteria: the StoppingCriteria-based abort hook for the HF
adapter.

How intervention timing actually gets enforced here (read this first)
-----------------------------------------------------------------------
`HFEngineAdapter.generate()` installs a forward hook on every extraction
point's configured layer/submodule (see `adapter.py`). Those hooks fire
DURING `model.forward()`, which HF's `generate()` loop calls once per decode
step, on the SAME thread that later calls this `StoppingCriteria`. Each hook
that matches a registered extraction point calls `router.route(record)`
synchronously and, for an inline extraction point whose signal came back
`action="abort"`, sets `state.should_stop = True` (see `_ActiveGeneration`
in `adapter.py`).

Crucially, `router.route()` ITSELF already enforces `InterventionPolicy`'s
`timeout_ms`/`on_timeout` for a `block_until_signal` extraction point (see
`undercurrent.router.Router._dispatch_with_intervention`) -- it never blocks the
calling thread (this one) longer than `timeout_ms` before returning either
the probe's real signal or the `on_timeout` fallback. So by the time
`model.forward()` returns and HF's decode loop reaches this class's
`__call__`, EVERY registered extraction point for this step has already been
given its bounded chance to intervene, one way or another. This class does
not need its own timeout logic -- it just has to read the flag the hook
already set. This is what "the StoppingCriteria check incorporates the
timeout fallback" means concretely: incorporation happens by construction,
because this whole call chain (hook -> route() -> StoppingCriteria) runs on
one thread, in one decode step, and `route()`'s own bound is what makes that
step's total added latency predictable.

This IS the "cleanest abort semantics" of any adapter in this platform,
mentioned elsewhere in this codebase: unlike a continuously-batched engine
(see `undercurrent.adapters.vllm`'s "INTERVENTION TIMEOUT LIMITATIONS"), "the
decode step" here really is the whole engine for this one request -- there
is no other concurrent request's forward pass this wait could stall. A
`block_until_signal` policy's blocking scope is exactly what the caller
asked for, with no caveats.

Internal: not part of the public API (see docs/api-stability.md) and may
change in any release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._optional import require_torch

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime torch dependency
    import torch

    from .adapter import _ActiveGeneration


class ProbingStoppingCriteria:
    """Duck-types `transformers.StoppingCriteria`: `__call__(input_ids,
    scores, **kwargs) -> torch.BoolTensor`.

    Does not subclass `transformers.StoppingCriteria` directly so this
    module stays importable without `transformers` installed (torch is
    still required at call time, via `require_torch()`, since the return
    value must be a real `torch.BoolTensor`) -- `transformers.generate()`
    only requires duck-typed callables in its `stopping_criteria` list, not
    a specific base class.
    """

    def __init__(self, state: _ActiveGeneration) -> None:
        self._state = state

    def __call__(self, input_ids: torch.Tensor, scores: Any, **kwargs: Any) -> torch.Tensor:
        torch = require_torch()
        batch_size = input_ids.shape[0]
        return torch.full((batch_size,), self._state.should_stop, dtype=torch.bool, device=input_ids.device)

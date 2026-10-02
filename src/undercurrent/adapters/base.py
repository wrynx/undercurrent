"""EngineAdapter: the abstract contract every inference-engine adapter
(HF, vLLM, llama.cpp, ...) implements for the activation-probing platform.

This module is the canonical home of the ABC. Every adapter
(`undercurrent.adapters.hf`, `undercurrent.adapters.vllm`, and any future
one) imports `EngineAdapter` from here rather than redefining it.

It deliberately imports nothing heavy (no torch, transformers or vllm), so
the contract stays importable -- and `import undercurrent` stays fast --
without pulling in an inference engine.

Contract (engine-agnostic -- every adapter must honor this regardless of how
its underlying engine batches, schedules, or represents sequences):

    load_model(model_name_or_path, **kwargs) -> None
        Load/initialize the underlying engine. Engine-specific kwargs (dtype,
        tensor_parallel_size, device, ...) pass straight through.

    register_extraction(request_id, extraction_points) -> None
        Wire up whatever capture mechanism the engine needs (hooks,
        callbacks, worker-side state, ...) for `request_id`, BEFORE that
        request's `generate()` call is made. Must not affect any other
        request_id's state -- concurrent requests' extraction points must
        stay isolated from one another.

    generate(request_id, prompt, generation_kwargs, router) -> str
        Run generation for `request_id`. For every token/layer combination
        that matches one of the request's registered extraction points, the
        adapter constructs an `ActivationRecord` and calls `router.route()`
        with it. If `router.route()` returns a `ProbeSignal` with
        `action == "abort"` for an inline extraction point, the adapter must
        stop generation -- as promptly as the underlying engine's execution
        model allows; see each adapter's own docs for the actual latency
        bound, since "promptly" means something very different for a single
        sequential decode loop (HF) than for a continuously-batched
        scheduler (vLLM). Returns the generated text.

        Intervention timing: `router.route()` itself already enforces
        whatever `InterventionPolicy` governs each extraction point (see
        `undercurrent.spec.InterventionPolicy` / `undercurrent.router.Router`) --
        `mode=reject` returns whatever `on_activation` produces with no
        additional wait; `mode=block_until_signal` blocks `route()`'s
        caller for at most `timeout_ms` before substituting the
        `on_timeout` fallback signal. An adapter's `generate()` does not
        need to (and should not) reimplement any of that timeout logic
        itself -- it only needs to call `route()` per matching activation
        and honor whatever `ProbeSignal` it gets back, exactly as it always
        has. What DOES vary per adapter is the *blocking scope* of that
        wait: for a single sequential decode loop, blocking is naturally
        scoped to just this one request's next token (see
        `undercurrent.adapters.hf.stopping_criteria`); for a continuously-batched
        engine, the same wait can widen to every request sharing that
        scheduler step (see `undercurrent.adapters.vllm.worker_extension`'s
        "INTERVENTION TIMEOUT LIMITATIONS" section) -- document that bound
        specifically in each concrete adapter, don't just inherit this
        docstring's generic language.

    unregister_extraction(request_id) -> None
        Tear down whatever `register_extraction` set up for `request_id`.
        Must be safe to call after `generate()` returns (normally or via
        abort), and must leave no per-request state behind that could leak
        into a future request reusing the same engine-internal slot.

What's deliberately NOT part of this contract, because it's engine-specific:
    - How positions are tracked internally (a simple running counter for a
      sequential engine; a scheduler-driven row/block mapping for a
      continuously-batched one).
    - Whether `generate()` is internally synchronous or drives an async
      event loop under the hood -- the external signature is synchronous
      either way.
    - Batching, multi-request concurrency mechanics, and how (or whether)
      hook state can be shared with a router instance across process
      boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotation-only; keeps this module import-light
    from ..router import Router
    from ..spec import ExtractionPoint


class EngineAdapter(ABC):
    """Abstract base every engine adapter implements.

    Implement it to probe an engine Undercurrent doesn't ship an adapter for.
    The contract is engine-agnostic: for each request, the caller calls
    ``register_extraction``, then ``generate``, then ``unregister_extraction``.

    During ``generate``, the adapter builds an
    [`ActivationRecord`][undercurrent.spec.ActivationRecord] for every
    token/layer that matches one of the request's extraction points and
    passes it to ``router.route(record)``. When ``route`` returns a signal
    with ``action == "abort"`` for an inline extraction point, the adapter
    stops generation as promptly as its engine allows. ``route`` already
    enforces each extraction point's
    [`InterventionPolicy`][undercurrent.spec.InterventionPolicy] (including
    ``block_until_signal`` timeouts), so adapters don't reimplement any
    timeout logic; they should document how widely that wait blocks (one
    request in a sequential decode loop, every request in the same scheduler
    step for a continuously batched engine).

    Requests must stay isolated from each other: registering, generating or
    unregistering one request must not affect another's state.
    """

    # Deliberately not part of the contract (engine-specific): how positions are
    # tracked internally, whether generate() drives an async event loop under the
    # hood (its signature is synchronous either way), and batching / concurrency /
    # cross-process mechanics.

    @abstractmethod
    def load_model(self, model_name_or_path: str, **kwargs: Any) -> None:
        """Load the underlying model/engine. Engine-specific kwargs (dtype, device, ...) pass through."""

    @abstractmethod
    def register_extraction(self, request_id: str, extraction_points: list[ExtractionPoint]) -> None:
        """Wire up capture for ``request_id``'s extraction points, before ``generate()`` is called for it.

        Must not affect any other request's state.
        """

    @abstractmethod
    def generate(self, request_id: str, prompt: str, generation_kwargs: dict[str, Any], router: Router) -> str:
        """Run generation for `request_id`, routing matching activations through `router`.

        Must honor an inline `ProbeSignal(action="abort")` by stopping
        generation as promptly as the underlying engine allows. Returns the
        generated text.
        """

    @abstractmethod
    def unregister_extraction(self, request_id: str) -> None:
        """Tear down whatever ``register_extraction`` set up for ``request_id``.

        Must be safe to call after ``generate()`` returns, normally or by abort,
        and must leave no state behind that could leak into a later request.
        """

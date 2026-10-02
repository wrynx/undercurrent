"""VLLMEngineAdapter: the `EngineAdapter` implementation for vLLM.

Ordering nuance vs. the HF adapter (read this before using the class)
-----------------------------------------------------------------------
The `EngineAdapter` contract says `register_extraction()` wires up capture
"before generate() is called." For vLLM, full wiring needs each request's
PROMPT LENGTH, because `SeqIdMapper.register_request()` (worker-side) must
know it before that request's very first (prefill) step -- and prompt length
isn't known until the prompt is tokenized, which this adapter doesn't do
until `generate()` starts. So: `register_extraction()` stores the
extraction points on the driver side (satisfying "call this before
generate()"); `generate()` tokenizes the prompt, then completes the actual
worker-side registration (extraction points + prompt_len together) before
submitting the request to the engine. A request that reaches the engine
without going through this path has no worker-side state and produces zero
ActivationRecords -- `seq_mapper.SeqIdMapper.resolve_step()` raises loudly
in that case rather than silently guessing a prompt length (see
seq_mapper.py).

Sync-looking API, async engine underneath
-------------------------------------------
vLLM's continuous batching -- and the concurrency this adapter needs to
support (multiple in-flight `generate()` calls actually sharing scheduler
steps, not just interleaved by the GIL) -- is only real when driven through
vLLM's async engine (`AsyncLLMEngine` / V1's `AsyncLLM`), whose `.generate()`
is an async generator. But `EngineAdapter.generate()`'s contract is a plain
synchronous method returning `str`. This adapter reconciles the two with the
standard "background event loop thread" pattern: `load_model()` starts one
dedicated event loop on its own thread; each synchronous `generate()` call
submits a coroutine onto that loop via `asyncio.run_coroutine_threadsafe(...)`
and blocks the CALLING thread (not the loop) on the result. Two Python
threads calling `adapter.generate(...)` concurrently therefore both actually
run on vLLM's shared engine loop concurrently, and vLLM's own scheduler is
free to batch their steps together -- which is exactly what the concurrency
integration test (`tests/test_integration_vllm.py`) exercises and what a
single sequential HF-style loop cannot demonstrate at all.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
from collections.abc import Callable
from typing import Any

from ...core import RequestContext
from ...errors import ProbingError
from ...router import Router
from ...spec import ActivationRecord, ExtractionPoint
from .._optional import COMPATIBILITY_DOC, require_torch, require_vllm
from ..base import EngineAdapter
from .version_check import check_vllm_version

logger = logging.getLogger(__name__)

_WORKER_EXTENSION_PATH = "undercurrent.adapters.vllm.worker_extension.ProbingWorkerExtension"


class VLLMAdapterLimitationError(ProbingError):
    """Raised for a documented, known limitation of the vLLM adapter.

    For example: an unsupported executor topology, or a vLLM API the adapter
    doesn't support. A distinct type, so callers can tell it from a bug.
    """


class VLLMEngineAdapter(EngineAdapter):
    """The [`EngineAdapter`][undercurrent.adapters.EngineAdapter] for vLLM (V1 engine), with concurrent ``generate()`` calls.

    Requires a supported vLLM version, checked on construction (vLLM itself
    is imported by ``load_model``). ``generate()`` is synchronous, but
    ``load_model`` starts vLLM's async engine on a background event-loop
    thread, so ``generate()`` calls from several threads run concurrently and
    share scheduler steps. Call
    [`shutdown`][undercurrent.adapters.vllm.VLLMEngineAdapter.shutdown] when
    done.

    ``register_extraction()`` only stores the extraction points; capture is
    set up in the worker when ``generate()`` has tokenized the prompt.

    ```python
    adapter = VLLMEngineAdapter()
    adapter.load_model("openai-community/gpt2")
    adapter.register_extraction(request_id, extraction_points)
    text = adapter.generate(request_id, prompt, {"max_tokens": 32}, router)
    adapter.unregister_extraction(request_id)
    ```

    See [The vLLM adapter](../internals/vllm-adapter.md) for how capture
    works and its limitations.
    """

    def __init__(self) -> None:
        # Fail fast on an unsupported vLLM version (see version_check.py). Reads
        # package metadata only; a missing vLLM is reported later by load_model().
        check_vllm_version()
        self._engine: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._tokenizer: Any = None
        self._pending_extraction_points: dict[str, list[ExtractionPoint]] = {}
        self._bound_router: Router | None = None
        self._model_name: str | None = None

    # -----------------------------------------------------------------
    # EngineAdapter contract
    # -----------------------------------------------------------------

    def load_model(self, model_name_or_path: str, **kwargs: Any) -> None:  # pragma: no cover - needs a live vLLM engine
        require_torch()
        require_vllm()
        from vllm import AsyncEngineArgs, AsyncLLMEngine  # local: only needed once vllm is confirmed present

        allow_unsupported_executor = kwargs.pop("allow_unsupported_executor", False)
        allow_v2_model_runner = kwargs.pop("allow_v2_model_runner", False)
        if not allow_v2_model_runner:
            # introspection.py (this adapter's row -> (request_id, token_pos) translation)
            # was written against vLLM's V1 GPUModelRunner's `input_batch` shape and doesn't
            # understand the V2 model runner's restructured internals (e.g.
            # model_runner.execute_model_state.input_batch, with renamed fields like
            # num_computed_tokens_np instead of num_computed_tokens_cpu) -- every V2 forward
            # hook fails loudly with VLLMIntrospectionError the moment capture is attempted.
            # Some installed vLLM versions/environments select V2 by default (or via an
            # already-set VLLM_USE_V2_MODEL_RUNNER) even for architectures that don't need it
            # (observed: plain gpt2, vLLM 0.28) -- force V1 here, the only model runner this
            # adapter's introspection layer supports, overriding any such default. Pass
            # allow_v2_model_runner=True to opt out (capture will then fail loudly at the
            # first forward hook until introspection.py is reconciled against V2's shape --
            # see that module's docstring).
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

        allow_request_id_randomization = kwargs.pop("allow_request_id_randomization", False)
        if not allow_request_id_randomization:
            # Some vLLM versions (observed: 0.28) rewrite the request_id you pass to
            # generate() before it ever reaches the scheduler -- InputProcessor.assign_request_id
            # replaces it with f"{your_request_id}-{8 random hex chars}" "to ensure uniqueness"
            # (see vllm.v1.engine.input_processor). This adapter's entire row ->
            # (request_id, token_pos) mapping depends on the request_id register_extraction()
            # was called with matching the request_id the scheduler/model runner report back in
            # scheduler_output/input_batch -- randomization breaks that silently until the first
            # forward hook fires, then fails loudly (SeqMapperError: "has a scheduled row range
            # but was never registered via register_request()", which kills the engine). Force
            # VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 here to keep request_id stable end-to-end,
            # unless the caller explicitly opts out via allow_request_id_randomization=True (in
            # which case capture fails loudly the same way, not silently). Vendor note: vLLM logs
            # this flag as deprecated ("will be removed in a future release") -- if a future vLLM
            # drops it entirely, this adapter has no correctness path without it and needs a
            # different fix (e.g. tracking vLLM's internal randomized ID back to ours some other
            # way) before it can support that version at all.
            os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
        engine_args = AsyncEngineArgs(
            model=model_name_or_path,
            worker_extension_cls=_WORKER_EXTENSION_PATH,
            **kwargs,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._model_name = model_name_or_path

        try:
            # _check_executor_topology() below needs to make an RPC call, which needs a running
            # loop to bridge onto (see _rpc()) -- must start this before that check, not after.
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, name="undercurrent-vllm-loop", daemon=True
            )
            self._loop_thread.start()

            self._check_executor_topology(allow_unsupported_executor)
        except BaseException:
            # Don't leave a half-started engine behind: its EngineCore process keeps GPU
            # memory, and garbage-collecting it later can block (see shutdown()).
            self.shutdown()
            raise

    def register_extraction(self, request_id: str, extraction_points: list[ExtractionPoint]) -> None:
        if request_id in self._pending_extraction_points:
            raise VLLMAdapterLimitationError(
                f"register_extraction(): request_id {request_id!r} is already pending registration "
                "(generate() hasn't been called for it yet). Call generate() for it, or use a fresh request_id."
            )
        self._pending_extraction_points[request_id] = list(extraction_points)

    def generate(self, request_id: str, prompt: str, generation_kwargs: dict[str, Any], router: Router) -> str:
        if self._engine is None or self._loop is None:
            raise VLLMAdapterLimitationError("generate() called before load_model(); call load_model(model) first")
        extraction_points = self._pending_extraction_points.pop(request_id, None)
        if extraction_points is None:
            raise VLLMAdapterLimitationError(
                f"generate({request_id!r}): no prior register_extraction() call for this request_id. "
                "Call register_extraction(request_id, extraction_points) first (ProbedModel does this for you)."
            )

        self._ensure_router_bound(router)

        prompt_token_ids = self._tokenize(prompt)
        prompt_len = len(prompt_token_ids)

        # Router needs register_request() before route() will accept anything for this
        # request_id -- it's what spawns the actual Probe instances. Every extraction point
        # of a request shares one RequestContext (mirrors undercurrent.router's own usage: it calls
        # `probe.on_start(request_ctx)` with the same object for every binding of a request,
        # not a per-extraction-point one), so extraction_point_config is left None here rather
        # than arbitrarily picking one extraction point to feature.
        request_ctx = RequestContext(
            request_id=request_id,
            prompt_metadata={"model": self._model_name, "prompt": prompt, "prompt_len": prompt_len},
            extraction_point_config=None,
        )
        router.register_request(request_id, extraction_points, request_ctx)
        # Pre-serialize to plain dicts before crossing the RPC boundary: ExtractionPoint is a
        # plain @dataclass, not a msgspec.Struct, so vLLM's typed-arg RPC decoder can't
        # reconstruct one from its encoded form and hands the worker a raw dict instead
        # (observed: vLLM 0.28, AttributeError: 'dict' object has no attribute 'tensor_type').
        # worker_extension.py's register_extraction()/_coerce_extraction_points() parses these
        # back via undercurrent.spec's own round-trip serializer.
        from undercurrent.spec import extraction_point_to_dict

        serialized_points = [extraction_point_to_dict(ep) for ep in extraction_points]
        self._rpc("register_extraction", args=(request_id, serialized_points, prompt_len))

        future = asyncio.run_coroutine_threadsafe(
            self._agenerate(request_id, prompt, generation_kwargs, router), self._loop
        )
        try:
            return future.result()
        finally:
            # Finalizes every probe for this request (on_end) and tears down router-side
            # state. Safe even if generation was aborted mid-stream -- see Router.end_request's
            # own docstring on why an aborted request's trajectory probe still gets a clean finish.
            router.end_request(request_id)

    def unregister_extraction(self, request_id: str) -> None:
        self._pending_extraction_points.pop(request_id, None)
        if self._engine is not None:
            self._rpc("unregister_extraction", args=(request_id,))

    def shutdown(self, timeout: float | None = 30.0) -> None:
        """Shut down the vLLM engine and the background event-loop thread.

        Not part of the ``EngineAdapter`` contract. Call it when tearing the
        adapter down; it's safe to call more than once, and the adapter can't
        be used afterwards.

        The engine is shut down explicitly, first: vLLM's `AsyncLLM` runs its
        `EngineCore` in a subprocess and talks to it over ZeroMQ. Leaving that
        to garbage collection keeps the subprocess (and its GPU memory) alive,
        and collecting the client later can block forever in ZeroMQ's
        `Context.term()` while its sockets are still open. `timeout` (seconds)
        bounds the engine shutdown.
        """
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                if "timeout" in inspect.signature(engine.shutdown).parameters:
                    engine.shutdown(timeout=timeout)
                else:  # older engines' shutdown() takes no timeout
                    engine.shutdown()
            except Exception:
                logger.warning("VLLMEngineAdapter.shutdown(): engine shutdown failed", exc_info=True)
        loop, thread = self._loop, self._loop_thread
        self._loop, self._loop_thread = None, None
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=10)
        if loop is not None and not loop.is_running() and not loop.is_closed():
            loop.close()
        self._bound_router = None
        self._pending_extraction_points.clear()

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _check_executor_topology(self, allow_unsupported_executor: bool) -> None:
        """Confirm there's exactly one worker before trusting capture.

        This used to read `self._engine.engine.model_executor` and check its
        class name for "UniProc". That doesn't work anymore on vLLM
        versions where `AsyncLLM`'s `EngineCore` (and therefore its
        `model_executor`) runs in a subprocess -- see this file's other
        version-seam comments -- so the attribute is simply absent on the
        driver side, and the old check failed closed (`<unknown>`,
        requiring `allow_unsupported_executor=True`) even for a genuine
        single-GPU setup that this adapter fully supports via
        `worker_extension.py`'s "CROSS-PROCESS ACTIVATION POLLING" fallback.

        What actually matters isn't the executor's class name, it's "is
        there exactly one worker to route activations through" -- Ray /
        tensor-parallel / multi-worker topologies would each need their own
        per-rank aggregation this adapter doesn't implement, regardless of
        which process(es) they run in. `collective_rpc` returns one result
        per worker by construction, so its length IS the worker count --
        ask the worker(s) directly rather than trying to introspect an
        attribute that may not be reachable at all.
        """
        try:
            results = self._rpc("pop_pending_aborts")  # harmless probe: no requests exist yet, always returns empty
        except Exception as exc:
            if allow_unsupported_executor:
                return
            raise VLLMAdapterLimitationError(
                f"could not determine worker topology (a probe RPC call raised {type(exc).__name__}: "
                f"{exc}) -- pass allow_unsupported_executor=True to load_model() to bypass this check "
                "if you're confident there's exactly one worker."
            ) from exc
        num_workers = len(results) if isinstance(results, list) else 1
        if num_workers <= 1 or allow_unsupported_executor:
            return
        raise VLLMAdapterLimitationError(
            f"load_model(): the vLLM engine runs {num_workers} workers (tensor or pipeline parallel), but the "
            "vLLM adapter is validated with exactly one. Use tensor_parallel_size=1 and "
            "pipeline_parallel_size=1, or pass allow_unsupported_executor=True (see below). "
            "Details: see "
            "worker_extension.py's 'PROCESS TOPOLOGY' / 'CROSS-PROCESS ACTIVATION POLLING' sections "
            "for why, and docs/_legacy/vllm_tp_pp_activation_extraction_notes.md for the tensor-parallel safety "
            "analysis. `_drain_pending_aborts`/`_drain_pending_activations` do merge every worker's "
            "results (not just rank 0) via `_merge_per_worker_results`, so single-PP-stage "
            "tensor-parallel topologies are no longer silently lossy -- but pipeline-parallel "
            "cross-rank ordering and Ray/distributed executors remain unvalidated. Pass "
            "allow_unsupported_executor=True to load_model() to bypass this check if you're confident "
            "your topology is one of the validated single-PP-stage cases."
        )

    @staticmethod
    def _merge_per_worker_results(pending: Any, key_fn: Callable[[Any], Any] | None = None) -> Any:
        """Flatten `collective_rpc`'s one-result-per-worker fan-out into a
        single deduplicated list, instead of just reading worker 0's list
        and discarding every other worker's.

        `pending` is either a flat single-worker result (nothing to merge --
        some vLLM/version code paths return the worker's own list directly
        rather than wrapping it in another list) or a genuine list of
        per-worker lists, one per rank. Only the latter shape needs merging;
        detected the same way the code this replaces did, by checking
        whether the first element is itself a list.

        Deduplication (via `key_fn`, or raw equality if omitted) matters
        because of what TP and PP each do to the same forward-hook event:
        - Under tensor parallelism, `RowParallelLinear`'s all-reduce (see
          `docs/_legacy/vllm_tp_pp_activation_extraction_notes.md`) means every TP rank
          computes an IDENTICAL full tensor for the same layer/token -- so
          every rank's hook fires and queues what is logically the same
          event. Without dedup, a TP degree of N would abort/route that one
          event N times.
        - Under pipeline parallelism, each rank owns disjoint layers, so
          different ranks' pending lists never share a key -- dedup is then
          a no-op and this naturally unions them instead, which is exactly
          "don't drop non-rank-0 workers' results."

        Not handled here: preserving true chronological order across
        DIFFERENT ranks' entries for the same request -- the result is
        grouped worker-by-worker. `_drain_pending_activations` re-sorts
        activation records into forward-pass order after merging (see its
        docstring); abort request_ids need no ordering.
        """
        if not isinstance(pending, list) or not pending or not isinstance(pending[0], list):
            return pending
        merged: list[Any] = []
        seen: set = set()
        for worker_result in pending:
            if not isinstance(worker_result, list):
                continue
            for item in worker_result:
                key = key_fn(item) if key_fn is not None else item
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def _ensure_router_bound(self, router: Router) -> None:
        """Try to hand the worker a live, in-process `Router` reference
        (the fast path: `route()` runs synchronously inside the forward
        hook, so an inline abort decision is available immediately). Falls
        back to cross-process activation polling -- see
        `worker_extension.py`'s 'CROSS-PROCESS ACTIVATION POLLING' -- when
        that's not possible, rather than raising, since this is now the
        EXPECTED case for some installed vLLM versions (observed: 0.28's
        `AsyncLLM`, which always runs `EngineCore` in a subprocess
        regardless of executor topology).

        The two ways `bind_router()` can fail, handled differently:
          - The RPC call itself raises before ever reaching the worker --
            vLLM's default (non-pickle) RPC encoder can't serialize a live
            `Router` (it owns a `ThreadPoolExecutor`, locks, spawned `Probe`
            instances) to cross a real process boundary. Caught broadly
            here (the exact exception type/message is a vLLM-version seam,
            not a stable contract to pattern-match on -- same reasoning as
            `_rpc`'s dual collective_rpc lookup) and treated as "fall back
            to polling," logged for visibility.
          - The RPC call succeeds but the worker's `bind_router()` itself
            returns `False` -- it ran, but detected (via its best-effort pid
            check) that it is NOT actually in-process despite the call
            having landed (e.g. some other cross-process transport that
            doesn't fail at the serialization step). This is different from
            the case above: something is inconsistent enough to warrant a
            loud failure rather than a silent fallback.
        """
        if self._bound_router is router:
            return
        if self._bound_router is not None:
            raise VLLMAdapterLimitationError(
                "a different Router instance was already bound to this adapter -- "
                "VLLMEngineAdapter supports exactly one Router for its whole lifetime. Pass the same Router to "
                "every generate() call, or create a new VLLMEngineAdapter for the other Router."
            )
        try:
            results = self._rpc("bind_router", args=(router,))
        except Exception as exc:
            logger.info(
                "undercurrent.adapters.vllm: could not hand the worker a live Router reference (%s: %s) -- "
                "falling back to cross-process activation polling. This is expected whenever the "
                "worker/EngineCore runs in a separate OS process from the driver, which some vLLM "
                "versions' AsyncLLM do unconditionally regardless of executor topology -- see "
                "worker_extension.py's 'CROSS-PROCESS ACTIVATION POLLING'.",
                type(exc).__name__,
                exc,
            )
            self._bound_router = router
            return
        ok = all(results) if isinstance(results, list) else bool(results)
        if not ok:
            raise VLLMAdapterLimitationError(
                "the vLLM worker extension reported it could not bind the Router in-process, so activations "
                "can't reach the probes. Please report this with your vLLM version and executor settings at "
                "https://github.com/wrynx/undercurrent/issues (background: ProbingWorkerExtension.bind_router(), "
                "'PROCESS TOPOLOGY')."
            )
        self._bound_router = router

    def _call_collective_rpc(self, method: str, args: tuple) -> Any:
        """Invoke `collective_rpc` via whichever entry point the installed
        vLLM exposes, returning whatever it returns as-is -- a plain value
        on some vLLM versions, an unawaited coroutine on others (see `_rpc`
        / `_rpc_async` for how each caller context resolves that).

        Version seam: V1's `AsyncLLM`/`AsyncLLMEngine` expose
        `engine.collective_rpc(...)` directly; some V0-era layouts route it
        through `engine.engine.model_executor.collective_rpc(...)` instead.
        Tries both; raises a clear reconciliation error if neither matches
        rather than failing with an opaque AttributeError deep in a hook.
        """
        engine = self._engine
        if hasattr(engine, "collective_rpc"):
            return engine.collective_rpc(method, args=args)
        inner_engine = getattr(engine, "engine", None)
        executor = getattr(inner_engine, "model_executor", None)
        if executor is not None and hasattr(executor, "collective_rpc"):
            return executor.collective_rpc(method, args=args)
        raise VLLMAdapterLimitationError(
            "could not find a collective_rpc() entry point on the loaded engine (tried "
            "engine.collective_rpc and engine.engine.model_executor.collective_rpc) -- the "
            "installed vLLM version's AsyncLLMEngine/AsyncLLM surface doesn't match what this "
            "adapter was written against. Reconcile VLLMEngineAdapter._rpc() against the "
            "installed vllm version before trusting registration/abort calls to reach the worker. "
            f"This usually means the installed vLLM is outside the supported range; see {COMPATIBILITY_DOC}."
        )

    def _rpc(self, method: str, args: tuple = ()) -> Any:
        """Call a `ProbingWorkerExtension` RPC method from the CALLING
        thread (i.e. NOT the adapter's own event-loop thread) -- used by
        `generate()`'s and `unregister_extraction()`'s synchronous call
        sites.

        Version seam: some installed vLLM versions' `collective_rpc` is a
        plain synchronous method; others (observed: `AsyncLLM` in vLLM
        0.28) declare it `async def`, in which case `_call_collective_rpc`
        returns an unawaited coroutine. Bounce that onto the adapter's own
        loop via `run_coroutine_threadsafe` and block the calling thread
        for the result -- safe here specifically because this method is
        never called from the loop thread itself (see `_rpc_async` for
        that case, which would deadlock if it tried this).
        """
        result = self._call_collective_rpc(method, args)
        if asyncio.iscoroutine(result):
            if self._loop is None:
                raise VLLMAdapterLimitationError(
                    f"_rpc({method!r}): collective_rpc returned a coroutine but no event loop is running"
                )
            return asyncio.run_coroutine_threadsafe(result, self._loop).result()
        return result

    async def _rpc_async(self, method: str, args: tuple = ()) -> Any:
        """Same as `_rpc`, but for callers already running ON the adapter's
        own event-loop thread (`_drain_pending_aborts`, called from inside
        `_agenerate`). Awaits an async `collective_rpc`'s coroutine
        directly instead of `_rpc`'s `run_coroutine_threadsafe(...).result()`
        bridge, which would deadlock the loop if used from here: it would
        block the very thread the scheduled coroutine needs to run on.
        """
        result = self._call_collective_rpc(method, args)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        engine = self._engine
        get_tok = getattr(engine, "get_tokenizer", None)
        if get_tok is not None:
            tok = get_tok()
            if asyncio.iscoroutine(tok):
                if self._loop is None:
                    tok.close()
                    raise VLLMAdapterLimitationError(
                        "_get_tokenizer(): engine.get_tokenizer() returned a coroutine but no event loop is running"
                    )
                tok = asyncio.run_coroutine_threadsafe(tok, self._loop).result()
            self._tokenizer = tok
            return tok
        inner = getattr(engine, "engine", None)
        tokenizer_group = getattr(inner, "tokenizer", None)
        tok = getattr(tokenizer_group, "tokenizer", tokenizer_group)
        if tok is None:
            raise VLLMAdapterLimitationError(
                "could not locate a tokenizer on the loaded engine (tried engine.get_tokenizer() "
                "and engine.engine.tokenizer.tokenizer) -- reconcile "
                "VLLMEngineAdapter._get_tokenizer() against the installed vllm version. "
                f"This usually means the installed vLLM is outside the supported range; see {COMPATIBILITY_DOC}."
            )
        self._tokenizer = tok
        return tok

    def _tokenize(self, prompt: str) -> list[int]:
        tokenizer = self._get_tokenizer()
        return tokenizer.encode(prompt)

    async def _agenerate(  # pragma: no cover - needs a live vLLM engine
        self, request_id: str, prompt: str, generation_kwargs: dict[str, Any], router: Router
    ) -> str:
        from vllm import SamplingParams

        sampling_params = SamplingParams(**generation_kwargs)
        final_text = ""
        async for request_output in self._engine.generate(prompt, sampling_params, request_id=request_id):
            await self._drain_pending_aborts()
            await self._drain_pending_activations(router)
            if request_output.outputs:
                final_text = request_output.outputs[0].text
            if request_output.finished:
                break
        return final_text

    async def _drain_pending_aborts(self) -> None:
        """Poll the worker for any request_ids an inline probe asked to
        abort since the last poll, and cancel them via the engine's own
        abort API.

        Timing caveat (documented, not a bug): vLLM schedules a whole step
        -- potentially many requests -- before checking what's finished or
        aborted when building the NEXT step. Calling `abort()` here cannot
        stop tokens already dispatched for the step that produced the
        `RequestOutput` we just consumed; the soonest an aborted request
        actually stops producing further tokens is the START of the next
        scheduler iteration. This is looser than a single sequential HF
        `generate()` loop, which can check a `StoppingCriteria` before every
        individual token. Expect at least one, and under heavier concurrent
        load possibly a few, extra tokens past the step whose activation
        triggered the abort -- `tests/test_integration_vllm.py`'s abort test
        asserts "stopped within a small bounded number of extra tokens," not
        "stopped at exactly N."
        """
        pending = await self._rpc_async("pop_pending_aborts")
        pending = self._merge_per_worker_results(pending)  # union across workers, not just rank 0 -- see docstring
        for rid in pending or []:
            abort_fn = getattr(self._engine, "abort", None)
            if abort_fn is None:
                logger.warning(
                    "undercurrent.adapters.vllm: engine has no abort() method; cannot honor abort for %r", rid
                )
                continue
            result = abort_fn(rid)
            if asyncio.iscoroutine(result):
                await result

    async def _drain_pending_activations(self, router: Router) -> None:
        """Route `ActivationRecord`s the worker buffered because no live
        `Router` reference ever reached its process -- see
        `worker_extension.py`'s 'CROSS-PROCESS ACTIVATION POLLING'. A no-op
        whenever `_ensure_router_bound()` succeeded in-process, since the
        worker's `_emit()` takes the synchronous fast path in that case and
        never populates its buffer.

        Routing runs off THIS adapter's own event loop, via
        `run_in_executor`: `Probe.on_activation` can do real work (a
        forward pass), and this same loop also drives every OTHER
        concurrently in-flight `generate()` call's communication with the
        engine core -- blocking it here would stall every concurrent
        request, not just this one. Same "threads, not asyncio" reasoning
        `undercurrent.router`'s own async worker pool is built on.

        Records are routed SEQUENTIALLY, in a well-defined order -- required,
        not just simpler: a trajectory probe's running state is only
        meaningful if its activations are fed to it in order, and routing
        them concurrently could reorder same-request records depending on
        which executor thread happens to finish first.

        Within a single worker's own list this is naturally its own capture
        order. But under pipeline parallelism, one `ExtractionPoint` can
        span layers owned by DIFFERENT workers (see
        `docs/_legacy/vllm_tp_pp_activation_extraction_notes.md`), and
        `_merge_per_worker_results` merges worker-by-worker -- it does not
        interleave by time. Left unsorted, that can hand a trajectory probe
        e.g. `tok1@L5, tok2@L5, tok1@L20, tok2@L20` instead of `tok1@L5,
        tok1@L20, tok2@L5, tok2@L20`. `token_pos` alone can't fix this since
        multiple layers share the same `token_pos` -- sorting by
        `(request_id, is_generated, token_pos, layer)` below restores the
        actual forward-pass order (prompt before generated tokens; ascending
        token position; and, within one position, ascending layer, since
        layer L's activation is always computed before layer L+1's for the
        same token).
        """
        pending = await self._rpc_async("pop_pending_activations")
        pending = self._merge_per_worker_results(
            pending,
            # Identity of a capture event, deliberately excluding the tensor
            # itself: TP replicates the SAME event (same request/extraction
            # point/layer/token) across every rank with an identical tensor
            # value, so this key is exactly what collapses those duplicates.
            key_fn=lambda raw: (
                raw["request_id"],
                raw["extraction_point_name"],
                raw["layer"],
                raw["token_pos"],
                raw["is_generated"],
            ),
        )
        if not pending:
            return
        pending = sorted(
            pending,
            key=lambda raw: (raw["request_id"], raw["is_generated"], raw["token_pos"], raw["layer"]),
        )
        loop = asyncio.get_running_loop()
        for raw in pending:
            record = ActivationRecord(
                request_id=raw["request_id"],
                extraction_point_name=raw["extraction_point_name"],
                layer=raw["layer"],
                token_pos=raw["token_pos"],
                tensor_type=raw["tensor_type"],
                tensor=raw["tensor"],
                is_generated=raw["is_generated"],
            )
            signal = await loop.run_in_executor(None, router.route, record)
            if signal is not None and signal.action.value == "abort":
                abort_fn = getattr(self._engine, "abort", None)
                if abort_fn is None:
                    logger.warning(
                        "undercurrent.adapters.vllm: engine has no abort() method; cannot honor abort for %r",
                        record.request_id,
                    )
                    continue
                result = abort_fn(record.request_id)
                if asyncio.iscoroutine(result):
                    await result

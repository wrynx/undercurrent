"""ProbingWorkerExtension: installed into vLLM's `Worker` via the
`worker_extension_cls` engine argument -- vLLM's supported mechanism for
mixing extra driver-callable methods and state into the worker process
(the same mechanism vLLM's own docs use for things like RLHF weight-sync
integrations). vLLM instantiates the class named by `worker_extension_cls`
and mixes it into the live `Worker` object via multiple inheritance, so
`self` here gets access to `self.model_runner` (and, through it,
`self.model_runner.model`, the actual `torch.nn.Module`) once the worker
has finished initializing -- NOT at `__init__` time, since the extension is
constructed as part of `Worker.__init__`, before `load_model()` has run.
Hook installation therefore happens lazily, on first use, not in `__init__`.

=====================================================================
 INTERCEPTION STRATEGY (option (a) from the task, chosen over (b))
=====================================================================
Two realistic options were considered:

  (a) A worker extension that installs `torch.nn.Module.register_forward_hook`
      on the underlying decoder layer / attention / MLP submodules (vLLM's
      model implementations are still ordinary torch.nn.Module graphs
      internally), then maps that step's scheduler metadata back to
      (request_id, token_pos) via `seq_mapper.py` / `introspection.py`.

  (b) vLLM's logits-processor / sampling-params extension points
      (`LogitsProcessor`, custom `SamplingParams` fields).

(b) was rejected: logits processors run once per request per step, AFTER
the full forward pass, on the FINAL logits tensor only. They never see
intermediate hidden states, attention output, or MLP output -- which is
exactly what `TensorType.RESIDUAL_STREAM` / `ATTN_OUT` / `MLP_OUT`
extraction points need -- and they have no natural per-layer hook point at
all (there's one set of logits per token, not one per (layer, tensor_type)).
They're also fundamentally single-request-scoped call sites, giving no
cleaner answer to the row-batch-to-request mapping problem than (a) does.
So (b) is a dead end for activation *capture*, though it's worth noting it
COULD have been a natural home for the *abort* signal specifically (a
logits processor can request a stop for its own request) -- this adapter
still uses `AsyncLLMEngine.abort()` directly for that instead (see
`adapter.py`), for uniformity with the same polling path used to plumb
aborts out of the worker process, rather than having two different
mechanisms for the same feature.

(a) is what's implemented here. It needs two additional pieces beyond
"call register_forward_hook":
  1. Per-step scheduler metadata (which request each row of the flattened
     batch tensor belongs to, and at what absolute position) -- vLLM
     doesn't hand this to a plain forward hook. We get it by wrapping
     `model_runner.execute_model` (see `_install_execute_model_wrapper`
     below) to snapshot `scheduler_output` for the duration of that call, so
     hooks fired during it can build/reuse a `StepBatchMetadata` (see
     `introspection.py`). This wrapper is a bound-method monkeypatch on the
     `model_runner` INSTANCE, not a vLLM-supported extension point --
     because vLLM's `worker_extension_cls` gives us a place to install
     hooks and RPC-callable methods, but no official "before/after one
     scheduler step" callback to hang metadata capture on. If a future vLLM
     version changes `execute_model`'s signature or stops calling it the
     way this assumes, this wrapper needs revisiting -- it's isolated here
     specifically so that's a small, findable diff.
  2. A channel for the RESULT (ActivationRecords, pending aborts) to get
     from the worker process back to the driver process where the Router
     and the adapter's `generate()` loop live. See "PROCESS TOPOLOGY" below.

=====================================================================
 PROCESS TOPOLOGY -- READ BEFORE USING WITH ANYTHING BUT A SINGLE
 LOCAL WORKER (vLLM's UniProcExecutor)
=====================================================================
`collective_rpc()` is driver-calls-worker: the engine's driver process
invokes a named method on the worker extension and blocks for its return
value. It is NOT designed for the worker to spontaneously push data to the
driver mid-forward-pass on its own initiative. That asymmetry matters here
because the driver's `Router` needs to receive every matching activation
AS IT'S CAPTURED (inline extraction points need `router.route()`'s return
value to decide whether to abort), and captures happen inside the worker's
forward pass.

  - `UniProcExecutor` (single local GPU/CPU device, no Ray, no
    multiprocessing) -- the ONLY topology this reference implementation
    fully supports for the hot path. The worker (and this extension) run
    in the SAME OS process as the driver/engine. `bind_router()` below is
    an ordinary Python method call handing this extension a direct
    reference to the live `Router` object (no serialization involved,
    despite going through `collective_rpc` for uniformity with the other
    RPC methods) -- so `route()` can be called straight from inside the
    forward hook. `adapter.py`'s `load_model()` checks the resolved
    executor class and raises `VLLMAdapterLimitationError` for anything
    else UNLESS the caller passes `allow_unsupported_executor=True`, so
    this limitation fails loudly rather than silently dropping
    activations.
  - Distributed executors (Ray, multiprocessing) run the worker in a
    SEPARATE process. `bind_router()` would need to pickle a `Router` --
    which owns a live `ThreadPoolExecutor`, locks, and spawned `Probe`
    instances -- across a process boundary, which does not work and isn't
    attempted. `adapter.py`'s `_ensure_router_bound()` catches the resulting
    RPC-serialization failure and falls back to CROSS-PROCESS ACTIVATION
    POLLING below rather than raising -- so this is no longer a hard
    failure, just a slower path.

Aborts, in contrast, only need a small SET of request_ids to cross the
worker -> driver boundary, not full tensors -- so that direction has always
used a plain polling RPC (`pop_pending_aborts`, called every step from
`adapter.py`) that works identically for either topology, whether or not
`bind_router()` succeeded.

=====================================================================
 CROSS-PROCESS ACTIVATION POLLING (bind_router() with no live Router)
=====================================================================
`UniProcExecutor` used to be a reliable signal that the worker and driver
share one OS process (see "PROCESS TOPOLOGY" above) -- true when this
adapter was first written. It no longer is: some installed vLLM versions'
`AsyncLLM` (observed: 0.28) run `EngineCore` in a dedicated subprocess
UNCONDITIONALLY, regardless of executor topology -- "EngineCore (starts the
engine in background process)" is vLLM's own comment on that call site. So
`bind_router()`'s RPC call can fail before it's even invoked: encoding a
live `Router` (owning a `ThreadPoolExecutor`, locks, spawned `Probe`
instances) to cross that boundary raises `TypeError: ... is not
serializable` in vLLM's own RPC layer. `adapter.py._ensure_router_bound()`
catches that and proceeds with `self._router` left `None` on this side --
`_emit()` below checks for exactly that, and instead of the synchronous
fast path (`self._router.route(record)`, immediate abort decision), buffers
the `ActivationRecord` into `self._pending_activations`.

`pop_pending_activations()` is the polling RPC the driver calls once per
step (same cadence as `pop_pending_aborts()`, from `adapter.py`'s
`_drain_pending_activations()`) to retrieve and clear that buffer, so
`Router.route()` itself runs in the DRIVER process -- where the real
`Router` lives -- instead of here. It returns plain dicts, not
`ActivationRecord` instances: vLLM's default (non-pickle) RPC encoder
natively supports `torch.Tensor` plus str/int/bool/float (see
`vllm.v1.serial_utils`'s `enc_hook`), but not arbitrary dataclass
instances, so `adapter.py` reconstructs `ActivationRecord`s from these dicts
on the receiving end.

Net effect versus the in-process fast path: activations (and therefore
abort decisions) lag by up to one polling interval instead of being
immediate, and per-record routing cost (e.g. an inline probe's forward
pass) moves from the worker's own thread to a thread on the driver's
executor (see `_drain_pending_activations()`'s docstring for why it must
NOT run on the driver's asyncio loop directly). Capture, position-matching,
and the seq_id -> (request_id, token_pos) translation are all unaffected --
only where `route()` executes changes.

=====================================================================
 INTERVENTION TIMEOUT LIMITATIONS (block_until_signal + continuous batching)
=====================================================================
`_emit()` below calls `self._router.route(record)` synchronously, from
inside the forward hook, when running the in-process fast path (see
"PROCESS TOPOLOGY" / "CROSS-PROCESS ACTIVATION POLLING" above -- under the
polling fallback, `route()` instead runs in the driver process, from
`adapter.py`'s `_drain_pending_activations()`). For an extraction point
whose effective `InterventionPolicy.mode` is `reject` (the default), that
is unchanged and safe: `route()` calls `on_activation` and returns whatever
it produces, with no additional wait-for-decision contract, same as before
this policy existed.

For `block_until_signal`, `Router.route()` DOES enforce `timeout_ms` --
this hook will never wait longer than that before getting a `ProbeSignal`
back (real or the `on_timeout` fallback), the same bounded-wait guarantee
every adapter gets from `undercurrent.router` for free. What is NOT enforceable
here, and cannot be fixed at this adapter's layer: the forward hook runs on
vLLM's single shared engine/scheduler thread, for a step that may batch
MANY different requests' tokens together (that's the entire point of
continuous batching). A bounded wait inside `route()` blocks that thread
for up to `timeout_ms` -- which stalls every OTHER request sharing that
step too, not just the one whose activation triggered the probe. Contrast
with a single sequential HF `generate()` loop (`undercurrent.adapters.hf`), where
"the decode step" IS the whole engine for that one request, so
`block_until_signal`'s blocking scope is exactly what the caller asked
for. Here it silently widens to "every concurrently-batched request," which
can violate other requests' latency expectations even though it never
violates the stated `timeout_ms` budget for the extraction point itself.

This is a fundamental mismatch between per-request synchronous gating and
this engine's batched execution model, not a bug this adapter can patch
around -- there is no way to make ONLY one row of a batched forward pass
wait without blocking the whole call, short of re-architecting capture to
run each request's forward pass independently (which would defeat
continuous batching's purpose entirely). The closest safe approximation
this adapter takes: it does not change `_emit()`'s call at all (the
router's own timeout budget and circuit breaker apply exactly as
documented), and `register_extraction()` below logs a loud one-time
warning the first time it sees an extraction point configured with
`mode=block_until_signal`, so an operator isn't silently surprised by
batch-wide stalls. Recommendation stated in that warning and in
the vLLM adapter design doc (docs/_legacy/): use `block_until_signal` with this adapter only for
single-request / effectively-unbatched deployments; prefer `reject` (or
`execution_mode=async`, i.e. observe-only) for anything with real
concurrent request batching. A router-level `default_intervention_policy`
of `block_until_signal` has the same effect but can't be detected from
here (this worker extension only ever sees the `ExtractionPoint` objects
themselves, never the Router's resolved per-binding policy) -- that case
is documented here and in the adapter's own docstring, not warned about at
runtime.

Internal and experimental: this reads undocumented vLLM internals and is
not part of the public API (see docs/api-stability.md); it changes whenever
vLLM does.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...core import ProbeAction
from ...errors import ProbingError
from ...router import Router
from ...spec import ActivationRecord, ExecutionMode, ExtractionPoint, InterventionMode, TensorType
from .adapter import VLLMAdapterLimitationError
from .introspection import check_introspection_compatible, extract_step_batch_metadata
from .seq_mapper import SeqIdMapper, StepBatchMetadata, TokenRowMapping
from .version_check import check_vllm_version

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime torch/vllm dependency
    import torch

logger = logging.getLogger(__name__)

_UNSUPPORTED_TENSOR_TYPES = (TensorType.KV,)


class WorkerExtractionError(ProbingError):
    """Raised for invalid worker-side extraction registration (e.g. an
    unsupported tensor_type like `kv`, or a duplicate/unknown request_id)."""


@dataclass
class _WorkerRequestState:
    extraction_points: tuple[ExtractionPoint, ...]


def _coerce_extraction_points(extraction_points: list[Any]) -> list[ExtractionPoint]:
    """Normalize `extraction_points` to real `ExtractionPoint` instances --
    see `register_extraction()`'s docstring for why they might instead be
    plain dicts.
    """
    if not extraction_points or not isinstance(extraction_points[0], dict):
        return list(extraction_points)
    from undercurrent.spec import parse_dict

    spec = parse_dict({"version": "1", "extraction_points": list(extraction_points)})
    return list(spec.extraction_points)


class ProbingWorkerExtension:
    """Mixed into vLLM's `Worker` via `worker_extension_cls`. See module docstring."""

    # Provided by the vLLM `Worker` this class is mixed into (an undocumented
    # internal, hence `Any`). Annotation only: no class attribute is created.
    model_runner: Any

    def __init__(self) -> None:
        self._ensure_state()

    def _ensure_state(self) -> None:
        """Initialize this extension's instance state, idempotently.

        `__init__` above is NOT a reliable place to do this: vLLM's
        `worker_extension_cls` mechanism mixes this class into `Worker` by
        mutating `worker_class.__bases__` to append it *after* `Worker` is
        already defined, then constructs `worker_class(**kwargs)` --
        `Worker.__init__`'s `super().__init__(...)` cooperative-inheritance
        chain terminates in `WorkerBase.__init__` without ever reaching
        this mixin's `__init__` (confirmed empirically against a real vLLM
        worker: a freshly constructed one has no `_requests` attribute at
        all, raising `AttributeError` the first time any RPC method here
        ran). So every RPC entry point below calls `_ensure_state()` first,
        instead of relying on `__init__` -- this guarantees state exists
        regardless of whether vLLM ever invokes this class's constructor.
        """
        if getattr(self, "_state_initialized", False):
            return
        # Here rather than in __init__ for the same reason as the state below.
        check_vllm_version()
        self._mapper = SeqIdMapper()
        self._requests: dict[str, _WorkerRequestState] = {}
        self._router: Router | None = None
        self._pending_aborts: set[str] = set()
        self._pending_activations: list[ActivationRecord] = []
        self._hooks_installed = False
        self._execute_model_wrapped = False
        self._introspection_checked = False
        self._current_step_batch: StepBatchMetadata | None = None
        self._current_step_mappings: list[TokenRowMapping] | None = None
        self._pending_scheduler_output: Any | None = None
        self._handles: list[Any] = []
        self._warned_block_until_signal = False
        self._state_initialized = True

    # -----------------------------------------------------------------
    # RPC methods -- called by the driver via engine.collective_rpc(...)
    # -----------------------------------------------------------------

    def bind_router(self, router: Router) -> bool:
        """Hand this worker extension a direct reference to the driver's
        Router. Only meaningful (and only returns True) when this extension
        is running in-process with the caller -- see "PROCESS TOPOLOGY"
        above. `adapter.py` is expected to check the return value and treat
        `False` as a hard configuration error, not a degraded mode.
        """
        import os

        self._ensure_state()

        # collective_rpc for a UniProcExecutor calls this method directly
        # (no pickling); for a distributed executor, `router` would have to
        # survive being pickled to get here at all, which a real Router
        # (thread pool, locks, live Probe instances) cannot do -- so by the
        # time we could plausibly answer "yes I'm in-process", a
        # multiprocessing/Ray topology has usually already failed to even
        # deliver this call. The pid comparison below is a best-effort
        # confirmation for the UniProcExecutor case, not a full detector.
        self._router = router
        self._driver_pid_at_bind = os.getpid()
        return True

    def register_extraction(self, request_id: str, extraction_points: list[Any], prompt_len: int) -> None:
        """Register `request_id`'s extraction points, before its `generate()` is submitted to the engine.

        `extraction_points` may arrive as real `ExtractionPoint` instances
        (direct in-process calls, e.g. unit tests) OR as plain dicts (it
        crossed a `collective_rpc` boundary): `undercurrent.spec.ExtractionPoint`
        is a plain `@dataclass`, not a `msgspec.Struct`, so vLLM's RPC
        layer's typed-arg reconstruction (`_convert_msgspec_args`, which
        tries to rebuild each arg into this method's own declared parameter
        type) can't turn its encoded form back into one -- it hands back a
        raw dict instead (observed: vLLM 0.28, `AttributeError: 'dict'
        object has no attribute 'tensor_type'` the moment this method tried
        to use one). `_coerce_extraction_points()` normalizes either shape
        via `undercurrent.spec`'s own round-trip parser -- `adapter.py`'s
        `generate()` pre-serializes with `extraction_point_to_dict()` for
        exactly this reason, so the common case is the dict path.
        """
        self._ensure_state()
        extraction_points = _coerce_extraction_points(extraction_points)
        if request_id in self._requests:
            raise WorkerExtractionError(
                f"register_extraction(): request_id {request_id!r} is already registered; use a fresh request_id"
            )
        for ep in extraction_points:
            if ep.tensor_type in _UNSUPPORTED_TENSOR_TYPES:
                raise WorkerExtractionError(
                    f"register_extraction(): extraction point {ep.name!r}: "
                    f"tensor_type={TensorType(ep.tensor_type).value!r} is not supported by the vLLM adapter -- "
                    "the KV cache lives in paged blocks addressed by slot mapping, not as a per-step "
                    "forward-pass tensor with one row per token the way residual_stream/attn_out/"
                    "mlp_out are, so it can't be captured through the same per-layer forward-hook "
                    "path. Use one of: "
                    + ", ".join(t.value for t in TensorType if t not in _UNSUPPORTED_TENSOR_TYPES)
                    + "."
                )
        self._requests[request_id] = _WorkerRequestState(extraction_points=tuple(extraction_points))
        self._mapper.register_request(request_id, prompt_len)
        self._ensure_hooks_installed()
        self._warn_if_block_until_signal(extraction_points)

    def _warn_if_block_until_signal(self, extraction_points: list[ExtractionPoint]) -> None:
        """See "INTERVENTION TIMEOUT LIMITATIONS" in this module's
        docstring: block_until_signal is not safe under real concurrent
        request batching with this adapter, since the forward hook's
        bounded wait runs on the shared engine thread and stalls every
        request in that scheduler step, not just the triggering one. Warns
        once per worker process (not once per request) so a long-running
        deployment isn't spammed."""
        if self._warned_block_until_signal:
            return
        for ep in extraction_points:
            policy = getattr(ep, "intervention", None)
            if policy is not None and policy.mode == InterventionMode.BLOCK_UNTIL_SIGNAL:
                logger.warning(
                    "undercurrent.adapters.vllm: extraction point %r is configured with "
                    "intervention.mode=block_until_signal. Under continuous batching, this "
                    "adapter's forward hook runs on the shared engine thread, so a bounded "
                    "wait for this probe's decision stalls every OTHER request sharing that "
                    "scheduler step, not just this one -- timeout_ms is still honored (route() "
                    "never blocks longer than that), but its blocking SCOPE silently widens to "
                    "the whole batch. Safe for single-request/effectively-unbatched "
                    "deployments only; prefer mode=reject (or execution_mode=async) for real "
                    "concurrent request batching. See worker_extension.py's module docstring, "
                    "'INTERVENTION TIMEOUT LIMITATIONS'.",
                    ep.name,
                )
                self._warned_block_until_signal = True
                return

    def unregister_extraction(self, request_id: str) -> None:
        """Tear down capture state for `request_id`. Idempotent."""
        self._ensure_state()
        self._requests.pop(request_id, None)
        self._mapper.unregister_request(request_id)
        self._pending_aborts.discard(request_id)

    def pop_pending_aborts(self) -> list[str]:
        """Return and clear the set of request_ids an inline probe asked to
        abort since the last call. Polled once per step by `adapter.py`,
        regardless of process topology (see module docstring)."""
        self._ensure_state()
        pending = list(self._pending_aborts)
        self._pending_aborts.clear()
        return pending

    def pop_pending_activations(self) -> list[dict[str, Any]]:
        """Return and clear `ActivationRecord`s buffered by `_emit()`
        because no live `Router` reference ever reached this process --
        see 'CROSS-PROCESS ACTIVATION POLLING' in this module's docstring.
        Always empty when `bind_router()` succeeded in-process, since
        `_emit()` takes the synchronous fast path in that case and never
        populates this buffer.

        Returns plain dicts, not `ActivationRecord` instances: vLLM's default
        RPC encoder supports arbitrary str/int/bool/float-keyed structures,
        but not arbitrary dataclass instances -- `adapter.py` reconstructs
        `ActivationRecord`s from these dicts on the driver side. `tensor` is
        sent as a plain nested list (`.tolist()`), not a raw `torch.Tensor`:
        the encoder's `torch.Tensor` support is a zero-copy, out-of-band
        buffer reference that only gets reconstructed at message positions
        its (typed-schema) encoding knows about ahead of time -- a Tensor
        nested inside our own ad-hoc dict-of-dicts doesn't qualify, and comes
        back on the driver side as the encoder's raw, un-reconstructed
        `(dtype, shape, buffer-ref)` triplet instead of a Tensor. A plain
        list needs no such reconstruction, at the cost of one extra copy
        (`to_flat_tensor` on the driver side accepts list input already).
        Polled once per step by `adapter.py`, same cadence as
        `pop_pending_aborts()`.
        """
        self._ensure_state()
        pending, self._pending_activations = self._pending_activations, []
        return [
            {
                "request_id": r.request_id,
                "extraction_point_name": r.extraction_point_name,
                "layer": r.layer,
                "token_pos": r.token_pos,
                "tensor_type": r.tensor_type,
                "tensor": r.tensor.tolist() if hasattr(r.tensor, "tolist") else r.tensor,
                "is_generated": r.is_generated,
            }
            for r in pending
        ]

    # -----------------------------------------------------------------
    # Hook installation (lazy: first call after the model is loaded)
    # -----------------------------------------------------------------

    def _ensure_hooks_installed(self) -> None:
        if self._hooks_installed:
            return
        decoder_layers = self._find_decoder_layers()
        for layer_idx, layer_module in decoder_layers.items():
            handle = layer_module.register_forward_hook(self._make_forward_hook(layer_idx, TensorType.RESIDUAL_STREAM))
            self._handles.append(handle)
            attn = getattr(layer_module, "self_attn", None)
            if attn is not None:
                self._handles.append(
                    attn.register_forward_hook(self._make_forward_hook(layer_idx, TensorType.ATTN_OUT))
                )
            mlp = getattr(layer_module, "mlp", None)
            if mlp is not None:
                self._handles.append(mlp.register_forward_hook(self._make_forward_hook(layer_idx, TensorType.MLP_OUT)))

        # The model-level final norm (e.g. Llama's trailing RMSNorm) runs
        # once, AFTER the last decoder layer -- it is a separate module, not
        # part of any decoder layer's own forward. A checkpoint trained on
        # `transformers`' `output_hidden_states=True` last tuple entry
        # (`outputs.hidden_states[-1]` / `outputs.last_hidden_state`) was
        # trained on THIS tensor, not on the raw last-decoder-layer residual
        # stream RESIDUAL_STREAM captures -- those differ by exactly this
        # norm. Tagged with layer=len(decoder_layers) (one past the last
        # decoder-layer index) since it has no natural per-decoder-layer
        # index of its own; a spec targets it with e.g. `layer: 32,
        # tensor: final_norm` for a 32-decoder-layer model.
        final_norm = self._find_final_norm()
        if final_norm is not None:
            final_norm_layer_idx = len(decoder_layers)
            handle = final_norm.register_forward_hook(
                self._make_forward_hook(final_norm_layer_idx, TensorType.FINAL_NORM)
            )
            self._handles.append(handle)

        self._install_execute_model_wrapper()
        self._hooks_installed = True
        logger.info("undercurrent.adapters.vllm: installed forward hooks on %d decoder layers", len(decoder_layers))

    def _find_decoder_layers(self) -> dict[int, torch.nn.Module]:
        """Locate the model's indexable decoder layer list.

        vLLM model implementations follow the same `model.model.layers[i]`
        convention as their HF counterparts (a `torch.nn.ModuleList` of
        decoder layers hung off the inner model). We walk a short list of
        plausible attribute paths rather than hardcoding one, and fail
        loudly (not silently skip hook installation) if none match --
        producing zero ActivationRecords for a registered extraction point
        would look like "the probe never fired" rather than "hook
        installation failed," which is a much worse failure mode to debug.
        """
        model = self.model_runner.model
        candidates = [
            getattr(getattr(model, "model", None), "layers", None),
            getattr(model, "layers", None),
            getattr(getattr(model, "transformer", None), "h", None),  # GPT-2-style vLLM impls
        ]
        for candidate in candidates:
            if candidate is not None and len(candidate) > 0:
                return dict(enumerate(candidate))
        raise WorkerExtractionError(
            "could not locate a decoder-layer list on the loaded model (checked "
            "model.model.layers, model.layers, model.transformer.h). This model's "
            "architecture isn't supported by undercurrent.adapters.vllm yet. Please report the model at "
            "https://github.com/wrynx/undercurrent/issues (the fix is adding its layer-list attribute "
            "path to ProbingWorkerExtension._find_decoder_layers())."
        )

    def _find_final_norm(self) -> torch.nn.Module | None:
        """Locate the model's model-level final norm module (e.g. Llama's
        trailing `RMSNorm`), applied once after the last decoder layer --
        see `_ensure_hooks_installed` for why this needs its own hook
        distinct from RESIDUAL_STREAM at the last decoder-layer index.
        Unlike `_find_decoder_layers`, returning `None` here (module not
        found) does not raise -- FINAL_NORM extraction points simply won't
        be captured for an unanticipated architecture, same fail-open
        behavior as a mistyped/unused tensor_type; only add a candidate
        attribute path here if you hit this for a real model.
        """
        model = self.model_runner.model
        candidates = [
            getattr(getattr(model, "model", None), "norm", None),
            getattr(model, "norm", None),
            getattr(getattr(model, "transformer", None), "ln_f", None),  # GPT-2-style vLLM impls
        ]
        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    def _install_execute_model_wrapper(self) -> None:
        """Monkeypatch `model_runner.execute_model` to snapshot `scheduler_output`
        for the duration of each step, so forward hooks fired during it can
        build this step's `StepBatchMetadata`. See "INTERCEPTION STRATEGY"
        in the module docstring for why this isn't a vLLM-supported hook
        point. Applied once; idempotent."""
        if self._execute_model_wrapped:
            return
        model_runner = self.model_runner
        original_execute_model = model_runner.execute_model

        def _wrapped_execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            if not self._introspection_checked:
                check_introspection_compatible(model_runner, scheduler_output)
                self._introspection_checked = True
            self._current_step_batch = None  # built lazily by the first hook that needs it this step
            self._current_step_mappings = None
            self._pending_scheduler_output = scheduler_output
            try:
                return original_execute_model(scheduler_output, *args, **kwargs)
            finally:
                self._current_step_batch = None
                self._current_step_mappings = None
                self._pending_scheduler_output = None

        model_runner.execute_model = _wrapped_execute_model
        self._execute_model_wrapped = True

    # -----------------------------------------------------------------
    # The forward hook itself
    # -----------------------------------------------------------------

    def _make_forward_hook(self, layer_idx: int, tensor_type: TensorType) -> Callable[..., None]:
        def _hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            if not self._requests:
                return  # fail open: no registered extraction points at all, nothing to do
            if not any(
                layer_idx in ep.layers and ep.tensor_type == tensor_type
                for state in self._requests.values()
                for ep in state.extraction_points
            ):
                return  # cheap short-circuit before touching the (possibly large) output tensor

            mappings = self._resolve_current_step_mappings()
            if mappings is None:
                logger.warning(
                    "undercurrent.adapters.vllm: forward hook fired for layer=%s tensor_type=%s outside a "
                    "tracked execute_model() call -- skipping capture for this call. This should not "
                    "happen in normal operation; see worker_extension.py's execute_model wrapper.",
                    layer_idx,
                    tensor_type,
                )
                return

            tensor = _extract_captured_tensor(output, tensor_type, layer_idx)

            for mapping in mappings:
                state = self._requests.get(mapping.request_id)
                if state is None:
                    continue
                for ep in state.extraction_points:
                    if layer_idx not in ep.layers or ep.tensor_type != tensor_type:
                        continue
                    if not ep.matches(
                        mapping.token_index,
                        mapping.is_generated,
                        self._mapper.prompt_len(mapping.request_id),
                        generated_index=mapping.generated_index,
                        # num_generated_total is intentionally omitted (None): resolving it would
                        # require knowing a request's FINAL generated length, which vLLM only
                        # reveals once a request finishes -- by which point the activation tensors
                        # for its earlier steps are long gone (they aren't retained past their own
                        # step; unlike HF's adapter, this one can't buffer a whole request's
                        # activations to re-check them at the end). Practical effect: a
                        # NEGATIVE-indexed single-point selector like `generated[-1]` will never
                        # match under this adapter (see PositionSelector.matches()'s own docstring:
                        # "the selector will not match until [num_generated_total] is supplied").
                        # `generated[*]` and `generated[start:]`/`generated[start:stop]` selectors
                        # are unaffected -- they don't need a final count. Documented limitation,
                        # not a silent gap: see the vLLM adapter design doc's "Limitations" (docs/_legacy/).
                        num_generated_total=None,
                    ):
                        continue
                    self._emit(mapping, ep, layer_idx, tensor_type, tensor)

        return _hook

    def _resolve_current_step_mappings(self) -> list[TokenRowMapping] | None:
        if self._current_step_mappings is not None:
            return self._current_step_mappings
        scheduler_output = self._pending_scheduler_output
        if scheduler_output is None:
            return None
        if self._current_step_batch is None:
            self._current_step_batch = extract_step_batch_metadata(self.model_runner, scheduler_output)
        self._current_step_mappings = list(self._mapper.resolve_step(self._current_step_batch))
        return self._current_step_mappings

    def _emit(
        self,
        mapping: TokenRowMapping,
        ep: ExtractionPoint,
        layer_idx: int,
        tensor_type: TensorType,
        tensor: torch.Tensor,
    ) -> None:
        import torch  # local import: only reached when a hook actually fires, i.e. torch is already loaded

        # float32, not the model's native bfloat16/float16: vLLM's RPC encoder
        # serializes tensors via numpy, which has no native bfloat16 support --
        # a bfloat16 tensor crossing collective_rpc (the cross-process
        # activation-polling fallback below) comes back as the encoder's raw
        # (dtype, shape, bytes) fallback tuple instead of a torch.Tensor, which
        # `to_flat_tensor` then fails to convert. The in-process fast path
        # (self._router is not None) never crosses that boundary, so it would
        # tolerate any dtype -- casting here keeps both paths identical.
        # copy=True: when the tensor is already float32 on CPU, `.to()` would
        # return a view, and fused-residual models' in-place add in the next
        # layer would then overwrite the captured values.
        row_tensor = tensor[mapping.row].detach().to("cpu", dtype=torch.float32, copy=True)
        record = ActivationRecord(
            request_id=mapping.request_id,
            extraction_point_name=ep.name,
            layer=layer_idx,
            token_pos=mapping.token_index,
            tensor_type=tensor_type.value if hasattr(tensor_type, "value") else tensor_type,
            tensor=row_tensor,
            is_generated=mapping.is_generated,
        )
        if self._router is not None:
            # In-process fast path: bind_router() handed us a live Router
            # reference -- route synchronously, exactly as this adapter has
            # always done, so an inline abort decision is available immediately.
            signal = self._router.route(record)
            if signal is not None and signal.action == ProbeAction.ABORT and ep.execution_mode == ExecutionMode.INLINE:
                self._pending_aborts.add(mapping.request_id)
            return
        # Cross-process fallback -- see 'CROSS-PROCESS ACTIVATION POLLING' in
        # this module's docstring. No live Router reference ever reached
        # this process, so buffer the record instead of dropping it;
        # adapter.py polls pop_pending_activations() every step and routes
        # these itself, in the driver process where the real Router lives.
        self._pending_activations.append(record)


def _extract_captured_tensor(output: Any, tensor_type: TensorType, layer_idx: int) -> torch.Tensor:
    """Pull the actual `[rows, hidden]` tensor out of a layer/submodule's
    forward-hook `output`.

    `residual_stream` hooks the whole decoder layer, whose output needs
    fused-residual handling: see `_residual_stream_from_layer_output()`.
    Every other tensor type hooks a submodule (`self_attn`, `mlp`, the final
    norm) whose output is either a bare tensor or a tuple whose first tensor
    is the one we want (e.g. the final norm's `(normed, residual)` pair). We
    take the first `Tensor` we find and REFUSE to guess further -- returning
    the wrong element of a tuple would silently produce a
    wrong-but-plausible-looking activation, which is worse than raising here.
    """
    import torch  # local import: only reached when a hook actually fires, i.e. torch is already loaded

    if tensor_type == TensorType.RESIDUAL_STREAM:
        return _residual_stream_from_layer_output(output, layer_idx)
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise WorkerExtractionError(
        f"forward hook for layer={layer_idx} tensor_type={tensor_type!r} got an output of type "
        f"{type(output).__name__} it doesn't know how to extract a tensor from. This model's decoder "
        "layer / self_attn / mlp forward signature wasn't anticipated -- extend "
        "_extract_captured_tensor() rather than letting this silently mis-capture."
    )


def _residual_stream_from_layer_output(output: Any, layer_idx: int) -> torch.Tensor:
    """The residual stream after decoder layer `layer_idx`, from that layer's
    forward output: the same quantity the HF adapter captures.

    vLLM decoder layers come in two shapes:

    - **Single tensor** (GPT-2's `GPT2Block`, Granite, ...): the layer adds
      the residual itself and returns the post-layer stream. Captured as is.
    - **Fused residual** (Llama, Mistral, Qwen2/3, Gemma 1/2/3, DeepSeek, ...):
      `forward(positions, hidden_states, residual)` returns
      `(hidden_states, residual)`, where `hidden_states` is the layer's MLP
      output (post-feedforward-normed on Gemma 2/3) and `residual` is the
      stream *before* that contribution is added. The add is deferred to the
      next layer's fused `input_layernorm(hidden_states, residual)`, or to the
      model's final norm, so the post-layer stream is
      `hidden_states + residual`. vLLM materialises it the same way
      (`EagleModelMixin._maybe_add_hidden_state`). A `(hidden_states, None)`
      pair (MiniCPM, FlexOlmo, Gemma 4) means the layer already added it.

    The sum is computed in the tensors' own dtype. vLLM's reference fused add
    computes in float32 and rounds back to that dtype; for fp16/bf16 inputs
    the two are bit-identical (float32 has enough spare precision that the
    double rounding is exact). The add is out of place, so the capture is
    fresh memory: vLLM's CUDA `fused_add_rms_norm` kernel overwrites both
    `hidden_states` and `residual` in place when the next layer runs, and
    that can't reach the captured tensor.

    Anything else (a 3-tuple, a second tensor of another shape or dtype)
    raises `VLLMAdapterLimitationError` rather than guessing which element,
    or which sum, is the residual stream.
    """
    import torch  # local import: only reached when a hook actually fires, i.e. torch is already loaded

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) == 2 and isinstance(output[0], torch.Tensor):
        hidden_states, residual = output
        if residual is None:
            return hidden_states
        if (
            isinstance(residual, torch.Tensor)
            and residual.shape == hidden_states.shape
            and residual.dtype == hidden_states.dtype
        ):
            return torch.add(hidden_states, residual)
    raise VLLMAdapterLimitationError(
        f"residual_stream at layer={layer_idx}: can't tell which part of this decoder layer's output is the "
        f"residual stream (got {_describe_layer_output(output)}). The vLLM adapter understands a single tensor "
        "and a fused-residual (hidden_states, residual) pair of matching shape and dtype. For this model, use "
        "attn_out, mlp_out or final_norm, or the HF backend, and please report it at "
        "https://github.com/wrynx/undercurrent/issues."
    )


def _describe_layer_output(output: Any) -> str:
    if isinstance(output, (tuple, list)):
        parts = [
            f"Tensor{tuple(item.shape)} {item.dtype}" if hasattr(item, "shape") else type(item).__name__
            for item in output
        ]
        return f"a {type(output).__name__} of {len(output)}: {', '.join(parts)}"
    return f"a {type(output).__name__}"

# undercurrent.adapters.vllm (legacy `probing_adapter_vllm` README)

The vLLM engine adapter for the activation-probing platform. Depends on
`undercurrent.spec` (`ExtractionPoint`, `ActivationRecord`, `PositionSelector`),
`undercurrent.core` (`Probe`, `ProbeSignal`), `undercurrent.router` (`Router`), and
`undercurrent.adapters.base` (`EngineAdapter`, the engine-agnostic ABC).

This is the **hardest** of the engine adapters, because vLLM's continuous
batching and paged attention mean there is no simple "token position N" the
way there is in a sequential HF `generate()` loop. Read "The core problem"
below before touching `seq_mapper.py` or `worker_extension.py`.

## The `EngineAdapter` contract

Every engine adapter (HF, vLLM, llama.cpp, ...) implements the same
engine-agnostic ABC:

```python
class EngineAdapter(ABC):
    def load_model(self, model_name_or_path: str, **kwargs) -> None: ...
    def register_extraction(self, request_id: str, extraction_points: list[ExtractionPoint]) -> None: ...
    def generate(self, request_id: str, prompt: str, generation_kwargs: dict, router: Router) -> str: ...
    def unregister_extraction(self, request_id: str) -> None: ...
```

**Provenance note:** this ABC lives in `undercurrent.adapters.base`, its
canonical home, and `undercurrent.adapters.vllm` imports (and re-exports) it
from there.

What's part of the contract (every adapter must honor this) vs. what's
vLLM-specific (this package's own problem to solve) is spelled out in
`undercurrent/adapters/base.py`'s docstring. Short version: the four method signatures,
the "one `ActivationRecord` per matching (token, layer)", and "an inline
`abort` signal must stop generation as promptly as the engine allows" are
the contract. *How* positions are tracked, whether `generate()` drives an
async engine under the hood, and how multi-request concurrency is
implemented are not.

## The core problem: there is no "token position N"

HF's `generate()` is one sequence, one loop, one tensor per step --
`token_pos` is just the loop counter. vLLM throws that away:

- Every decoder-layer forward pass operates on ONE flattened tensor of
  shape `[total_tokens_in_step, hidden]`, concatenating whatever mix of
  prefill chunks and single decode tokens the scheduler picked for *this
  step* across potentially many different requests.
- Which row-range belongs to which request, and at what absolute position
  in that request's own sequence, changes every step: requests join and
  leave the batch independently, a finished request's row slot gets reused
  by a newly admitted one, and long prompts can be chunked across several
  prefill steps before decode even starts.

So "row → (request_id, token_pos)" has to be recomputed from that step's
scheduler metadata, every step. That's `seq_mapper.SeqIdMapper`.

## Interception strategy: worker extension + forward hooks, not logits processors

Two options were on the table:

- **(a) — chosen.** A worker extension (vLLM's `worker_extension_cls`
  plugin mechanism) that installs `torch.nn.Module.register_forward_hook`
  on the underlying decoder layer / `self_attn` / `mlp` submodules (vLLM
  model implementations are still ordinary `torch.nn.Module` graphs
  internally), then maps that step's scheduler metadata back to
  `(request_id, token_pos)` via `seq_mapper.py` / `introspection.py`.
- **(b) — rejected.** vLLM's logits-processor / `SamplingParams` extension
  points. These run once per request per step, *after* the full forward
  pass, on the final logits tensor only -- they never see intermediate
  hidden states, attention output, or MLP output, which is exactly what
  `residual_stream` / `attn_out` / `mlp_out` extraction points need. No
  natural per-layer hook point exists there at all. Dead end for capture,
  though it's worth noting a logits processor *could* have been a home for
  the abort signal specifically -- this adapter uses `AsyncLLMEngine.abort()`
  directly instead, for uniformity with the same worker→driver polling path
  used to plumb activations out.

Full writeup, including the two extra pieces (a) needs beyond "call
`register_forward_hook`" -- per-step scheduler metadata capture via an
`execute_model` wrapper, and the worker↔driver channel problem -- is in
`worker_extension.py`'s module docstring.

## Process topology: `UniProcExecutor`, with a cross-process fallback

`collective_rpc()` is driver-calls-worker; it isn't designed for a worker to
push data to the driver on its own initiative. That matters because the
driver's `Router` needs every matching activation *as it's captured* (inline
extraction points need `route()`'s return value to decide whether to
abort), and captures happen inside the worker's forward pass.

- **`UniProcExecutor`** (single local device, no Ray, no multiprocessing) --
  the only fully-supported topology. This *used to* reliably mean "worker
  and driver share one OS process," letting `bind_router()` hand the worker
  extension a direct Python reference to the live `Router` (no
  serialization) and route synchronously from inside the forward hook. It
  no longer does on every vLLM version: some installed versions' `AsyncLLM`
  (observed: 0.28) run `EngineCore` in a dedicated **subprocess**
  unconditionally, regardless of executor topology -- so `bind_router()`'s
  RPC call itself fails (`TypeError: ... is not serializable`, since a live
  `Router` owns a `ThreadPoolExecutor` and locks that can't cross that
  boundary) before ever reaching the worker.

  `VLLMEngineAdapter._ensure_router_bound()` catches that failure and falls
  back to **cross-process activation polling** instead of raising:
  `ProbingWorkerExtension._emit()` buffers each `ActivationRecord` it
  captures (as a plain, msgspec-safe dict, not the dataclass itself --
  vLLM's RPC encoder natively supports `torch.Tensor` plus
  str/int/bool/float, not arbitrary class instances) instead of routing it
  in-process; `VLLMEngineAdapter._drain_pending_activations()` polls
  `pop_pending_activations()` once per step (same cadence as
  `pop_pending_aborts()`, from `_agenerate()`) and calls `router.route()`
  itself, in the driver process where the real `Router` lives -- via
  `run_in_executor`, so a probe's own compute doesn't stall the driver's
  shared event loop (which also carries every OTHER concurrently in-flight
  request's communication with the engine core). Net effect versus the
  in-process fast path: activations, and therefore abort decisions, lag by
  up to one polling interval instead of being immediate. Whichever path is
  active is decided once per adapter lifetime and is transparent to
  callers -- `generate()`'s contract is unchanged either way. See
  `worker_extension.py`'s "CROSS-PROCESS ACTIVATION POLLING" for the full
  design.
- **Distributed executors** (Ray, multiprocessing -- i.e. more than one
  actual model-forward worker) are a different case from the above and
  remain genuinely unsupported: `collective_rpc` would call every worker
  and each would need its own aggregated activation stream, which this
  adapter doesn't implement. `VLLMEngineAdapter._check_executor_topology()`
  used to check the resolved executor's class name for `UniProcExecutor` --
  but that reads `self._engine.engine.model_executor`, which (like
  `bind_router`'s live `Router` reference) isn't reachable from the driver
  side at all on a vLLM version whose `AsyncLLM` subprocesses `EngineCore`,
  so the check used to fail closed even for a genuine single-GPU setup this
  adapter fully supports. It now asks the worker(s) directly instead:
  `collective_rpc` returns one result per worker by construction, so a
  cheap probe RPC's result length reveals the real worker count regardless
  of which process(es) they run in. `load_model()` raises
  `VLLMAdapterLimitationError` for more than one worker (or if the probe
  call itself fails), unless you explicitly pass
  `allow_unsupported_executor=True` (which silently drops activations from
  workers other than rank 0 instead of routing them -- documented, not
  hidden).

Aborts are the one thing that has always crossed the worker → driver
boundary via polling regardless of topology: only a small *set* of
request_ids needs to travel, so `pop_pending_aborts()` works exactly as it
did before this fallback existed.

## Abort / intervention wiring

When an inline extraction point's probe returns `ProbeSignal(action="abort")`,
the worker extension adds the request_id to a pending-aborts set
(`ProbingWorkerExtension._pending_aborts`). `VLLMEngineAdapter._agenerate()`
polls `pop_pending_aborts()` after every `RequestOutput` it consumes and
calls the engine's own cancellation API (`AsyncLLMEngine.abort(request_id)`
/ the equivalent on whatever version is installed) for anything returned.

**Timing caveat, by design, not a bug:** vLLM schedules a whole step --
potentially many requests -- before checking what's aborted when building
the *next* step. Calling `abort()` cannot stop tokens already dispatched for
the step whose activation triggered the abort; the soonest a request
actually stops is the start of the next scheduler iteration. This is looser
than a single sequential HF loop, which can check a `StoppingCriteria`
before every individual token. Expect at least one, and under heavier
concurrent load possibly a few, extra generated tokens past the triggering
step. `tests/adapters/vllm/test_integration_vllm.py`'s abort test asserts "stopped well
short of `max_tokens`," not "stopped at exactly N."

## `InterventionPolicy.mode=block_until_signal` is unsafe here under real concurrency

`_emit()` in `worker_extension.py` calls `router.route(record)` exactly as
it always has, and `Router.route()` itself already enforces
`timeout_ms`/`on_timeout` for a `block_until_signal` extraction point (see
the router's legacy guide in `docs/_legacy/`) -- this hook never waits longer than
`timeout_ms` for a `ProbeSignal`. What can't be fixed at this adapter's
layer: the forward hook runs on vLLM's single shared engine/scheduler
thread, for a step that may batch MANY different requests' tokens together
(continuous batching's entire point). A bounded wait there stalls that
whole step -- every OTHER request sharing it, not just the one whose
activation triggered the probe. Contrast with `undercurrent.adapters.hf`, where
"the decode step" IS the whole engine for one request, so the same policy's
blocking scope is exactly what was asked for.

This is a fundamental mismatch between per-request synchronous gating and
this engine's batched execution model, not a bug. The closest safe
approximation taken here: `_emit()`'s call is unchanged (the router's own
timeout budget and circuit breaker apply exactly as documented), and
`ProbingWorkerExtension.register_extraction()` logs a loud one-time warning
the first time it sees an extraction point configured with
`mode=block_until_signal` (see `worker_extension.py`'s "INTERVENTION
TIMEOUT LIMITATIONS" section). **Use `block_until_signal` with this adapter
only for single-request / effectively-unbatched deployments; prefer
`mode=reject` (or `execution_mode=async`, i.e. observe-only) for anything
with real concurrent request batching.** A router-level
`default_intervention_policy=block_until_signal` has the identical effect
but can't be detected/warned about from here -- this worker extension only
ever sees the `ExtractionPoint`s themselves, never the Router's resolved
per-binding policy.

## Known limitations (documented, not silent)

- **Distributed executors (Ray / multiprocessing, i.e. more than one
  model-forward worker) aren't supported** -- see "Process topology" above.
  `load_model()` fails loudly rather than silently dropping activations,
  unless explicitly overridden. (This is unrelated to the cross-process
  *activation-polling* fallback also described there, which handles a
  single `UniProcExecutor` worker whose vLLM version puts `EngineCore` in
  its own subprocess -- still one real worker, just not in the driver's OS
  process.)
- **`tensor_type: kv` extraction points are rejected at registration.** KV
  cache lives in paged blocks addressed by slot mapping, not as a per-step
  forward-pass tensor with one row per token the way
  `residual_stream`/`attn_out`/`mlp_out` are. Capturing it needs
  block-table-aware logic this adapter doesn't implement.
  `ProbingWorkerExtension.register_extraction()` raises
  `WorkerExtractionError` for it instead of silently mis-mapping.
- **Negative-indexed single-point selectors (`generated[-1]`, etc.) never
  match.** Resolving them needs a request's *final* generated length, known
  only once vLLM reports the request finished -- by which point that
  step's activation tensors are long gone (unlike HF's adapter, this one
  can't buffer a whole request's activations to re-check them at the end).
  `generated[*]` and slice selectors (`generated[5:]`, `generated[2:8]`)
  are unaffected. See the `matches()` call site in `worker_extension.py`.
- **`introspection.py` reads undocumented vLLM internals**
  (`model_runner.input_batch.req_ids` / `.num_computed_tokens_cpu`,
  `scheduler_output.num_scheduled_tokens`) inferred from vLLM's V1
  architecture, not a stable public API. No vLLM install was available in
  the environment this adapter was authored in to pin an exact version
  against. `check_introspection_compatible()` fails loudly (not silently)
  if the installed vLLM's shape doesn't match -- see that module's
  docstring for what to check first if you hit it.
- **Only vLLM's V1 model runner is supported, not V2.** Some vLLM versions
  ship a second, substantially restructured "V2" model runner
  (`vllm.v1.worker.gpu.model_runner`) with a different internal batch shape
  entirely (e.g. `model_runner.execute_model_state.input_batch` instead of
  `model_runner.input_batch`, `num_computed_tokens_np` instead of
  `num_computed_tokens_cpu`) that `introspection.py` doesn't understand at
  all. Some installed vLLM versions/environments select V2 by default (or
  via an already-set `VLLM_USE_V2_MODEL_RUNNER`) even for architectures that
  don't need it -- observed against plain `gpt2` on vLLM 0.28, which broke
  capture entirely (`VLLMIntrospectionError` at the very first forward
  hook). `VLLMEngineAdapter.load_model()` now forces `VLLM_USE_V2_MODEL_RUNNER=0`
  before building the engine, unless you pass `allow_v2_model_runner=True`
  (in which case capture fails loudly with `VLLMIntrospectionError` instead
  of silently mis-mapping activations, same as every other version seam in
  this file). Full V2 support isn't implemented.
- **Some vLLM versions randomize `request_id` before it reaches the
  scheduler.** vLLM 0.28's `InputProcessor.assign_request_id` rewrites the
  `request_id` you pass to `generate()` into
  `f"{your_request_id}-{8 random hex chars}"` "to ensure uniqueness" --
  which breaks this adapter's entire row → `(request_id, token_pos)`
  mapping: `register_extraction()` registers under the plain `request_id`
  you gave it, but the scheduler/model runner then report the *randomized*
  one back, so the first forward hook fails loudly
  (`SeqMapperError: ... has a scheduled row range but was never registered`,
  which kills the engine). `load_model()` now forces
  `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` before building the engine to
  keep `request_id` stable end-to-end, unless you pass
  `allow_request_id_randomization=True` (capture then fails loudly the same
  way, not silently). vLLM logs this flag itself as deprecated -- if a
  future vLLM version drops it entirely, this adapter has no correctness
  path without it and will need a different fix (e.g. tracking vLLM's
  internal randomized ID back to the caller's some other way) before it can
  support that version.
- **`adapter.py`'s `_rpc()` / `_get_tokenizer()` try a couple of known
  attribute paths** (`engine.collective_rpc` vs.
  `engine.engine.model_executor.collective_rpc`; `engine.get_tokenizer()`
  vs. `engine.engine.tokenizer.tokenizer`) across vLLM's V0/V1 API
  differences, and raise `VLLMAdapterLimitationError` if neither matches,
  rather than an opaque `AttributeError` three frames into a hook.
- **The `execute_model` wrapper is a bound-method monkeypatch**, not a
  vLLM-supported extension point -- `worker_extension_cls` gives a place to
  install hooks and RPC methods, but no official "before/after one
  scheduler step" callback to hang per-step metadata capture on. See
  `worker_extension.py`'s `_install_execute_model_wrapper` docstring.
- **`ExtractionPoint` args are pre-serialized to plain dicts before crossing
  `collective_rpc`.** `ExtractionPoint` is a plain `@dataclass`, not a
  `msgspec.Struct`, so some vLLM versions' typed-arg RPC decoder can't
  reconstruct one from its encoded form and hands the worker a raw dict
  instead (observed: vLLM 0.28, `AttributeError: 'dict' object has no
  attribute 'tensor_type'` the first time `register_extraction` tried to use
  one). `adapter.py`'s `generate()` serializes with `undercurrent.spec`'s own
  `extraction_point_to_dict()`; `worker_extension.py`'s
  `register_extraction()`/`_coerce_extraction_points()` parses them back via
  `parse_dict()` on the receiving end -- the same round-trip serializer
  `undercurrent.spec` already tests for spec YAML round-tripping, just reused
  across a process boundary. Also accepts real `ExtractionPoint` instances
  unchanged (e.g. direct in-process/unit-test calls), so this is transparent
  either way.

## Testing

```
pip install -e ".[dev]"          # at the repo root
python -m pytest tests/adapters/vllm
```

54 tests run with **no torch or vLLM installed at all**:

- `test_seq_mapper.py` -- the seq_id → `(request_id, token_pos)` translation
  core, in isolation, against hand-built `StepBatchMetadata` (mocked
  scheduler metadata): single-shot prefill, chunked prefill, decode-step
  advancement, two concurrent requests sharing one step, a finished
  request's row slot reused by a new one, a request dropping out of the
  batch for a step and resuming later.
- `test_worker_extension_bookkeeping.py` -- registration/unregistration,
  duplicate-request and `kv`-rejection, hook installation reaching a fake
  model (idempotent across requests), decoder-layer-discovery fallback and
  failure, `pop_pending_activations()` drain/clear semantics.
- `test_worker_extension_capture.py` -- end-to-end hook-firing →
  row-mapping → `ActivationRecord` → `Router.route()` → abort-signal, via a
  minimal fake `torch` module installed into `sys.modules` (a standard
  trick for exercising import-guarded code without the real dependency).
  Includes the two-concurrent-requests-in-one-step case at the unit level,
  and the cross-process fallback (`_emit()` buffering instead of routing
  when no `Router` is bound).
- `test_adapter_contract.py` -- `VLLMEngineAdapter`'s ordering contract
  (`register_extraction` before `generate`), the `_rpc`/topology-check
  version-seam fallback logic, router-binding rules (including the
  cross-process fallback when `bind_router`'s RPC call itself raises), and
  `_drain_pending_activations()`'s record-reconstruction/routing/abort
  behavior.

`test_integration_vllm.py` needs a GPU and a real vLLM install: it is marked
`gpu` (auto-skipped without CUDA by the root `tests/conftest.py`) and is
additionally gated behind `pytest.importorskip("vllm")`. **It was not run in the sandbox this adapter
was authored in** (no GPU, no vLLM install available there) -- see that
file's module docstring. It covers: a `single_shot` extraction point
producing one correct record, a `generated[*]` trajectory extraction point
firing across multiple decode steps, that same case with a *second*
concurrent request in flight (the scenario HF's adapter can't exercise at
all), and an inline abort actually halting generation early.

## Usage

```python
from undercurrent.adapters.vllm import VLLMEngineAdapter
from undercurrent.router import Router

adapter = VLLMEngineAdapter()
adapter.load_model("gpt2", gpu_memory_utilization=0.3, max_model_len=64)

router = Router(probe_registry={...})
adapter.register_extraction(request_id, extraction_points)
text = adapter.generate(request_id, prompt, {"max_tokens": 32, "temperature": 0.0}, router)
adapter.unregister_extraction(request_id)

adapter.shutdown()  # stops the background event loop thread; not part of the ABC
```

Note the ordering nuance vs. HF: `register_extraction()` stores the
extraction points on the adapter (satisfying "call this before
`generate()`"), but the *worker-side* registration (which needs the
prompt's tokenized length) only completes once `generate()` starts. See
`adapter.py`'s module docstring for why.

## `vllm serve` CLI path: `general_plugins` entry point

Everything above is the *embedded* usage path: you construct
`VLLMEngineAdapter` yourself and call `load_model()`, which builds
`AsyncEngineArgs` itself and can pass `worker_extension_cls=` directly.
That's not the only way to run vLLM against this adapter -- you can also
run a plain `vllm serve <model> ...` CLI process, provided you wire two
things in yourself:

1. **`--worker-extension-cls undercurrent.adapters.vllm.worker_extension.ProbingWorkerExtension`**
   (or the equivalent key in a `--config some.yaml` vLLM already supports).
   **This is still required and is not automatic.** We verified against the
   installed vLLM 0.28.0 source that a `vllm.general_plugins` entry point
   function is called with zero arguments, in every process vLLM starts
   (API-server/frontend, `EngineCore`, and every TP/PP worker subprocess),
   and is never handed a config object to mutate -- by the time any plugin
   runs, `EngineArgs.worker_extension_cls`'s default is already fixed for a
   bare `vllm serve` invocation. There is no supported way to set it from a
   plugin. If you skip this flag, `vllm serve` runs fine but captures
   nothing.
2. Installing `undercurrent`, which registers itself as a
   `vllm.general_plugins` entry point (`undercurrent =
   "undercurrent.adapters.vllm.plugin:register"` in the root `pyproject.toml`) so vLLM
   loads `undercurrent/adapters/vllm/plugin.py`'s `register()` automatically on
   `vllm serve` startup. Scoped deliberately modestly: it logs that the
   plugin loaded (with a best-effort guess at which process it's running
   in, for log readability only), and set-if-unset's the same two
   correctness-critical env vars `adapter.py`'s `load_model()` forces for
   the embedded path (`VLLM_USE_V2_MODEL_RUNNER=0`,
   `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`) -- see `plugin.py`'s module
   docstring for exactly why a zero-arg, multiply-invoked hook is a safe
   place to do that (and why `worker_extension_cls`/config mutation is not).

An OpenAI-compatible probing server is on the roadmap; until then,
`examples/openai_server/serve_llama_mlp_pipeline.py` is a working reference server.

# Tensor-parallel / pipeline-parallel activation extraction: research notes

Research-only findings for extending `undercurrent.adapters.vllm` (formerly `probing_adapter_vllm`) beyond
`UniProcExecutor` (single process/GPU). No source in this repo was
modified for this investigation. All citations are against an installed
vLLM 0.28.0 source tree:

```
<site-packages>/vllm/   (vLLM 0.28.0)
```

Adapter-side citations are against
`src/undercurrent/adapters/vllm/worker_extension.py` and
`adapter.py` in this repo.

## What the adapter currently hooks, per `tensor_type`

From `ProbingWorkerExtension._ensure_hooks_installed()`
(`src/undercurrent/adapters/vllm/worker_extension.py:431-466`):

| `tensor_type`      | Hooked module                                   | Citation |
|---|---|---|
| `residual_stream`  | the decoder layer itself (`LlamaDecoderLayer`)  | `worker_extension.py:436` |
| `attn_out`         | `layer.self_attn` (`LlamaAttention`)            | `worker_extension.py:438-440` |
| `mlp_out`          | `layer.mlp` (`LlamaMLP`)                        | `worker_extension.py:441-443` |
| `final_norm`       | the model-level final `RMSNorm`                 | `worker_extension.py:456-462` |

All of these are `register_forward_hook` calls, which fire **after** the
target module's `forward()` has fully returned, with `output` being
whatever that `forward()` returned (`(module, input, output)`).

## Question 1 — Tensor parallelism, per `tensor_type`

### The key mechanic: `RowParallelLinear` all-reduces *inside* `forward()`, before returning

`vllm/model_executor/layers/linear.py:1635-1661` (the `RowParallelLinear.forward`
body — class starts at `linear.py:1504`):

```python
def forward(self, input_):
    ...
    output_parallel = self.quant_method.apply(self, input_parallel, bias_)
    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)   # <- all-reduce HERE
    else:
        output = output_parallel
    ...
    return output, output_bias
```

The all-reduce (`tensor_model_parallel_all_reduce`) happens as a statement
*inside* `forward()`, and its result (`output`) is what gets returned. A
`torch.nn.Module.register_forward_hook` fires strictly after `forward()`
returns its value — there is no way for a hook, wherever it is attached in
the module tree, to observe the pre-all-reduce `output_parallel`. This
means the TP-safety question isn't actually "which submodule is the hook
on" — it's simply "does `reduce_results=True` for this linear," because by
the time *any* hook fires (on the linear itself or on any ancestor module
that calls it), the reduce has already completed.

### `o_proj` (attention output projection)

`vllm/model_executor/models/llama.py:173-179` (`LlamaAttention.__init__`):

```python
self.o_proj = RowParallelLinear(
    input_size=self.total_num_heads * self.head_dim,
    output_size=hidden_size,
    bias=bias_o_proj,
    quant_config=quant_config,
    prefix=f"{prefix}.o_proj",
)
```

`reduce_results` is **not passed**, so it takes `RowParallelLinear`'s
default of `True` (`linear.py:1547`). So `o_proj`'s `forward()` always
all-reduces internally when `tp_size > 1` (the reduce is gated only on
`self.reduce_results and self.tp_size > 1`, `linear.py:1653-1654`).

`LlamaAttention.forward()` (`llama.py:222-232`) does:

```python
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)   # o_proj already all-reduced internally
return output
```

**Verdict for `attn_out`** (hooked on `self_attn`, i.e. `LlamaAttention`):
**FULL, replicated tensor**, identical on every TP rank. A hook placed
directly on `o_proj` instead would see exactly the same thing, for the
reason above — the reduce is inside `o_proj.forward()` itself, not
something that happens later in `self_attn.forward()`. There is no
"pre-reduce partial result" observable via a forward hook anywhere in this
call chain under vLLM's default config.

### `down_proj` (MLP output projection)

`vllm/model_executor/models/llama.py:80-120` (`LlamaMLP`):

```python
def __init__(self, ..., reduce_results: bool = True, ...):
    ...
    self.down_proj = RowParallelLinear(
        input_size=intermediate_size,
        output_size=hidden_size,
        bias=bias,
        quant_config=quant_config,
        reduce_results=reduce_results,
        disable_tp=disable_tp,
        prefix=f"{prefix}.down_proj",
    )
```

`LlamaDecoderLayer.__init__` constructs `self.mlp = LlamaMLP(...)`
(`llama.py:298-305`) **without** passing `reduce_results`, so it also
defaults to `True`.

**Verdict for `mlp_out`** (hooked on `mlp`, i.e. `LlamaMLP`): **FULL,
replicated tensor**, same reasoning as `attn_out` — `down_proj`'s internal
all-reduce completes before `LlamaMLP.forward()` (`llama.py:116-120`)
returns, and before the `mlp` forward hook fires.

**Caveat worth flagging in the doc, not a change to make now:** this
depends on `reduce_results` staying at its default `True` for both
`o_proj` and `down_proj`. Nothing in vLLM prevents a model implementation
or quantization path from constructing these with `reduce_results=False`
(e.g. for a fusion optimization that defers the reduce to a later fused
kernel) — the plain Llama path checked here does not do that, but this
should be re-verified per model architecture, not assumed universal.

### `residual_stream`

`LlamaDecoderLayer.forward()` (`llama.py:311-328`):

```python
hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
return hidden_states, residual
```

By the time this returns, `self_attn(...)` and `mlp(...)` have already
produced full/replicated outputs (per above), and `RMSNorm` and the
residual-add (`post_attention_layernorm`, `input_layernorm`) are pure
elementwise/replicated ops with no sharding at all — every TP rank holds
the complete `hidden_size` dimension for `hidden_states`/`residual`
throughout. **Verdict: FULL, replicated tensor**, unconditionally,
regardless of `tp_size`.

Note: the adapter's hook fires on the whole decoder layer and receives the
`(hidden_states, residual)` tuple as `output` — see
`_extract_captured_tensor()` in `worker_extension.py:660+` for how it picks
one of the two.

### `final_norm`

`vllm/model_executor/models/llama.py:389-392`:

```python
if get_pp_group().is_last_rank:
    self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
else:
    self.norm = PPMissingLayer()
```

`RMSNorm` has no tensor-parallel sharding at all — it's a plain elementwise
op over the full `hidden_size` on whichever rank runs it. **Verdict: FULL,
replicated tensor** on every TP rank within the PP stage that holds it
(see Question 2 for which PP rank that is).

### Summary table — Question 1

| `tensor_type` | Hooked module | TP-safe (full tensor)? |
|---|---|---|
| `residual_stream` | decoder layer | Yes — always full, no sharding at this point in the graph |
| `attn_out` | `self_attn` (`LlamaAttention`) | Yes — `o_proj` is `RowParallelLinear(reduce_results=True)` by default; all-reduce completes inside `o_proj.forward()`, so any hook downstream of it (including on `o_proj` itself) sees the reduced result |
| `mlp_out` | `mlp` (`LlamaMLP`) | Yes — `down_proj` is `RowParallelLinear(reduce_results=True)` by default, same reasoning |
| `final_norm` | model's final `RMSNorm` | Yes — no sharding; only present on the last PP rank (see Q2) |

**Bottom line for TP:** for stock Llama with default `reduce_results=True`
on `o_proj`/`down_proj` (true of every example spec in
`examples/content_safety/*.yaml` — none override this), a
`register_forward_hook`-based capture is TP-safe for all four
`tensor_type`s used today: every rank's hook fires with the identical,
full-`hidden_size` tensor. The remaining TP engineering problem is not
about capturing the *right value* — it's that under a multi-worker
executor, every TP rank runs its own `ProbingWorkerExtension` and would
each independently emit an `ActivationRecord` for the same
`(layer, tensor_type, token_pos)`, so the driver-side aggregation needs to
either only route rank 0's copy or explicitly dedupe (see
`VLLMEngineAdapter._check_executor_topology()`,
`adapter.py:130-146ish`/README's "Process topology" section, which
currently hard-fails for >1 worker rather than doing this).

## Question 2 — Pipeline parallelism: layer routing

Layer partitioning across PP ranks happens in
`vllm/model_executor/models/utils.py:786-818` (`make_layers`), which
`LlamaModel.__init__` calls at `llama.py:384-388`:

```python
start_layer, end_layer, modules = make_layers(...)
```

Inside `make_layers`:

```python
from vllm.distributed.utils import get_pp_indices
start_layer, end_layer = get_pp_indices(
    num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
)
modules = torch.nn.ModuleList(
    [PPMissingLayer() for _ in range(start_layer)]
    + get_offloader().wrap_modules(
        layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
    )
    + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
)
```
(`vllm/model_executor/models/utils.py:802-816`)

`get_pp_indices` (`vllm/distributed/utils.py:127-167`) computes an even
(or `VLLM_PP_LAYER_PARTITION`-overridden) split of `num_hidden_layers`
across `pp_size` ranks and returns this rank's `[start_layer, end_layer)`
range.

Critically, layer indices **outside** a given rank's range are filled with
`PPMissingLayer()` placeholders (`utils.py:773-783`), a bare
`torch.nn.Identity` subclass whose own `forward()` just passes its input
through. `self.model.layers` therefore has length `num_hidden_layers` on
*every* PP rank, but only the slice `[start_layer, end_layer)` holds real
`LlamaDecoderLayer` instances.

`LlamaModel.forward()` (`llama.py:401-440`) only ever calls the layers in
that local range:

```python
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    hidden_states, residual = layer(positions, hidden_states, residual, **extra_layer_kwargs)
```
(`llama.py:421-426`)

So `PPMissingLayer` placeholders for out-of-range indices are never
invoked at all during a forward pass on that rank.

Same story for `final_norm`: `self.norm` is a real `RMSNorm` only
`if get_pp_group().is_last_rank`, else a `PPMissingLayer()`
(`llama.py:389-392`), and non-last ranks return an `IntermediateTensors`
before ever reaching `self.norm` (`llama.py:431-434`).

**Adapter interaction:** `ProbingWorkerExtension._find_decoder_layers()`
(`worker_extension.py:468-494`) walks `model.model.layers` and enumerates
*all* `num_hidden_layers` entries — including the `PPMissingLayer`
placeholders on ranks that don't own them — and calls
`register_forward_hook` on every single one indiscriminately (no
PP-awareness in this code path today). This is harmless, not a bug: since
`register_forward_hook` just attaches a callback that fires on the next
`forward()` call, and `PPMissingLayer.forward()` is simply never invoked
by `LlamaModel.forward()`'s ranged loop, hooks on out-of-range layers
(and, on non-last ranks, the `final_norm` hook on a `PPMissingLayer`
`self.norm`) are installed but permanently dormant on that rank.

**Confirmed:** a `worker_extension_cls` instance on a given PP rank only
ever sees forward calls (and therefore only ever fires hooks) for the
decoder layers physically present on that rank. A spec's `layers: 20`
extraction point on a rank that only holds layers 0-15 installs a hook
that is syntactically valid but never fires — it does not error, and it
does not silently pick up some other layer's data; it simply never
produces an `ActivationRecord`. (This means a naive multi-PP-rank rollout
needs each rank's worker extension to report which layers it actually
covers, or the driver has no way to distinguish "extraction point never
matched because layer is on a different PP stage" from "no such token
processed yet" — currently indistinguishable failure modes from the
`ActivationRecord` stream alone.)

## Question 3 — Cross-rank abort propagation

### The abort call chain

`AsyncLLM.abort()` (`vllm/v1/engine/async_llm.py:749-761`):

```python
async def abort(self, request_id, internal: bool = False) -> None:
    request_ids = ...
    all_request_ids = self.output_processor.abort_requests(request_ids, internal)
    await self.engine_core.abort_requests_async(all_request_ids)
```

This routes (via `core_client.py`, whichever client class is active for
the installed topology) down to `EngineCore.abort_requests()`
(`vllm/v1/engine/core.py:484-490`):

```python
def abort_requests(self, request_ids: list[str]):
    """Abort requests from the scheduler."""
    self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
```

This is the entire mechanism: aborting a request means telling the
**single, centralized `Scheduler`** instance (`vllm/v1/core/sched/scheduler.py:69`,
`class Scheduler`) inside `EngineCore` to mark it finished. There is
exactly one `Scheduler`/`EngineCore` for the whole distributed deployment
— it is not per-PP-rank or per-TP-rank. Both step execution and abort
enforcement go through this one scheduler.

### Why the abort itself needs no extra propagation once it reaches the scheduler

Every model-forward step, regardless of topology, is issued the same way:
`Executor.execute_model()` fans a single `SchedulerOutput` out to **every**
worker via `collective_rpc(...)`
(`vllm/v1/executor/multiproc_executor.py:337-348`):

```python
def execute_model(self, scheduler_output, non_block=False):
    return self.collective_rpc("execute_model", args=(scheduler_output,), ...)
```

And `MultiProcExecutor.collective_rpc()`
(`vllm/v1/executor/multiproc_executor.py:372-411`) broadcasts to every
worker process over a shared broadcast message queue
(`self.rpc_broadcast_mq`) and gathers one result per worker — this
broadcast covers all TP × PP ranks uniformly; there is no separate
propagation step per PP stage. Because the *same* `SchedulerOutput` object
(built once, centrally, from the single `Scheduler`'s state) is what's
broadcast, once `scheduler.finish_requests()` has marked a request
`FINISHED_ABORTED`, the very next `SchedulerOutput` simply omits that
request's tokens for every rank — first, second, and last PP stage alike
— without any additional cross-rank signaling. So: **once an abort reaches
the (single, centralized) scheduler, it is automatically enforced on all
PP stages as part of vLLM's normal step loop.** No extra plumbing is
needed on vLLM's side for that direction.

### Where extra plumbing IS needed: the *discovery* path, worker → driver, under multi-worker topologies

The vLLM-side mechanism above only helps once the driver has actually
called `engine.abort(request_id)`. Getting a **probe-discovered** abort
signal (found inside a forward hook on some worker) back to the driver is
this adapter's own problem, not vLLM's — and the current implementation
is explicitly single-worker-shaped:

`ProbingWorkerExtension._pending_aborts` is a `set()` local to each
worker's own process
(README "Abort / intervention wiring" section;
`worker_extension.py` maintains it per-instance). The driver polls it via
`collective_rpc("pop_pending_aborts")`
(`adapter.py:233` for the topology probe call, and the real per-step poll
at `adapter.py:436`/`adapter.py:473`):

```python
pending = await self._rpc_async("pop_pending_aborts")
if ...:
    pending = pending[0]  # collective_rpc returns one result per worker; normalize the common single-worker case
```
(`adapter.py:436-438`, and again at `adapter.py:473`)

`collective_rpc` mechanically *does* reach every worker across every PP
(and TP) rank and returns a list with one entry per worker — so if a
`ProbingWorkerExtension` on an intermediate PP stage added a request_id to
its own local `_pending_aborts` upon seeing (e.g.) a `residual_stream`
match at a layer that stage owns, that worker's contribution **would**
come back inside the `pending` list from `collective_rpc`. But
`adapter.py` currently does `pending = pending[0]` — it unconditionally
takes only the first worker's (rank 0's) result and discards the rest.
Under the currently-enforced single-worker topology
(`VLLMAdapterLimitationError` for >1 worker, see
`_check_executor_topology()` per the README) this is a no-op
simplification, not a bug. But it is exactly the "additional plumbing"
this task asked about: **for a PP topology with more than one worker
process, an abort discovered on a non-rank-0 PP stage would currently be
silently dropped by this line** — it never reaches
`engine.abort_requests_async(...)`, because the adapter only looks at
`pending[0]`. Fixing this is small in principle (union all workers'
returned sets instead of indexing `[0]`) but is exactly the kind of
per-worker-aggregation logic the README already flags as unimplemented for
multi-worker executors generally (activations have the identical
"only rank 0 is read, others silently dropped" issue when
`allow_unsupported_executor=True` is set — see README "Process topology").

### Summary — Question 3

- **Abort enforcement across PP stages, once the driver calls
  `engine.abort()`:** automatic, no extra plumbing — a single centralized
  `Scheduler` drives one `SchedulerOutput` broadcast to all ranks every
  step, so an aborted request simply stops being scheduled everywhere,
  immediately as of the next step (same one-step latency documented in the
  README for the already-supported single-worker case).
- **Getting a PP-rank-local probe-discovered abort signal to the driver in
  the first place:** `collective_rpc`'s broadcast mechanism already
  reaches every rank's worker process and could carry every rank's pending
  aborts back, but this adapter's current polling code
  (`adapter.py`'s `pending = pending[0]`) only reads worker index 0's
  result. This must change (aggregate/union across all returned
  per-worker lists) before an intermediate-PP-stage-discovered abort can
  reach the driver under any multi-worker topology.

## Recommended milestone-1 scope

Given the above:

1. **TP alone (`tensor_parallel_size > 1`, `pipeline_parallel_size == 1`),
   any of the four already-supported `tensor_type`s
   (`residual_stream`, `attn_out`, `mlp_out`, `final_norm`)**: the
   per-rank tensor each hook sees is already full/replicated for stock
   Llama with default `reduce_results=True` (Question 1) — the only real
   engineering gap is driver-side deduplication, since every TP rank's
   worker extension independently captures and would independently emit
   the *same* value for a given `(request_id, layer, tensor_type,
   token_pos)`. This is a bounded, well-understood fix (route/emit from
   one designated rank only, e.g. TP rank 0 — cheap because the tensor
   content doesn't differ by rank) and does not require touching
   `seq_mapper.py`'s per-step row-mapping logic at all, since that logic
   is per-step scheduler metadata, which vLLM already keeps identical
   across TP ranks within one PP stage. **Recommend this as the actual
   milestone-1 target**: single PP stage, TP ranks >1, restrict initial
   support to `residual_stream` and `final_norm` first (simplest — no
   per-linear-layer config to re-verify across model architectures/
   quantization paths), then extend to `attn_out`/`mlp_out` once the
   `reduce_results=True` assumption has been spot-checked against
   whatever quantized/fused kernel paths are in scope (the plain-Llama
   check here does not cover MoE or fused-QKVO kernel variants).
2. **PP alone**: layer-to-rank routing is fully deterministic and
   introspectable (`get_pp_indices`, `vllm/distributed/utils.py:127-167`),
   and out-of-range hooks are provably inert rather than incorrect
   (Question 2) — so PP support is mostly a driver-side aggregation
   problem (collecting `ActivationRecord`s that now arrive from different
   worker processes covering disjoint layer ranges, and running
   `abort_requests_async` reachability described in Question 3) rather
   than a correctness risk on the capture side. Still recommend deferring
   PP to milestone 2: it requires collective_rpc result aggregation across
   *disjoint* layer coverage (not just TP's "same value from every rank"
   dedup), and the abort-discovery `pending[0]` fix described above is a
   hard prerequisite for inline (abort-capable) extraction points under
   PP — observe-only (`execution_mode=async`) extraction points could
   ship slightly earlier since they don't depend on the abort path at all.
3. **TP × PP combined**: defer until both of the above are independently
   solid — it's the union of both engineering gaps (per-rank dedup *and*
   disjoint-layer-range aggregation *and* the abort `pending[0]` fix), with
   no new correctness risk beyond what's listed above, but the most
   surface area to get the driver-side aggregation wrong on.
4. **Out of scope regardless of topology** (unaffected by this
   investigation, already called out in the adapter's README): `kv`
   tensor_type (paged/block-table-addressed, no per-step per-token
   tensor), V2 model runner, negative-indexed single-point selectors.

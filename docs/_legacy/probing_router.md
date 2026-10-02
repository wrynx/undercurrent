# probing_router

The **router** for the activation-probing platform: dispatches
`ActivationRecord`s to the correct probe instance(s), per request,
according to each extraction point's `execution_mode`.

Built on [`undercurrent.spec`](../../src/undercurrent/spec/) (`ExtractionPoint`,
`ActivationRecord`) and [`undercurrent.core`](../../src/undercurrent/core/) (`Probe`,
`ProbeFactory`/`.spawn()`, `ProbeSignal`, `ProbeResult`) — types are
imported directly from those packages, never redefined.
Implements no inference-engine adapter: the router is fully testable using
synthetic `ActivationRecord` streams, with no real inference engine
involved.

## Install

```bash
pip install -e .   # at the repo root: the undercurrent project (spec, core, router)
```

Requires Python 3.9+.

## Usage

```python
from undercurrent.router import ProbeAction, ProbeFactory, Router
from undercurrent.core import RequestContext

router = Router(probe_registry={
    "linear_probe": ProbeFactory(MyClassifierProbe),
    "trajectory_probe": ProbeFactory(MyTrajectoryProbe, {"threshold": 0.8}),
})

router.register_request(
    request_id,
    extraction_points,   # list[undercurrent.spec.ExtractionPoint]
    RequestContext(request_id, prompt_metadata, extraction_point_config),
)

for record in activation_stream:                 # synthetic or adapter-fed
    signal = router.route(record)
    if signal is not None and signal.action is ProbeAction.ABORT:
        break                                     # inline probes only: adapter's cue to stop generation

results = router.end_request(request_id)          # {extraction_point_name: ProbeResult}
router.shutdown()                                  # once, when the router itself is torn down
```

`probe_registry` maps `ExtractionPoint.probe_type` to the `ProbeFactory`
that should be spawned for it — the router has no way to know which `Probe`
subclass a `probe_type` string refers to otherwise, so this mapping is
owned and supplied by the caller (e.g. application startup config).

## `Router` responsibilities

- **`register_request(request_id, extraction_points, request_ctx)`** —
  spawns one fresh probe instance per extraction point via
  `ProbeFactory.spawn()`, calls `probe.on_start(request_ctx)`, and stores
  everything keyed by `(request_id, extraction_point_name)`. Validates all
  extraction points before spawning any probe (all-or-nothing), and
  defensively re-rejects `execution_mode=async` + `probe_kind=single_shot`
  even though `undercurrent.spec`'s parser should already have caught it — the
  router doesn't trust that every `ExtractionPoint` it's handed came
  through that path.
- **`route(record)`** — looks up the probe registered for
  `(record.request_id, record.extraction_point_name)`, defensively checks
  the record's `layer`/`tensor_type` are consistent with that extraction
  point's config, then dispatches:
  - **inline, `intervention.mode=reject`** (the default): calls
    `on_activation` synchronously and returns whatever `ProbeSignal` it
    produced immediately — this is what an engine adapter uses to decide
    whether to abort generation. No wait-for-decision contract beyond
    whatever `on_activation` itself naturally takes.
  - **inline, `intervention.mode=block_until_signal`**: waits up to
    `timeout_ms` for that same call, substituting `on_timeout`'s fallback
    `ProbeSignal` if it doesn't finish in time — see
    [Intervention policy](#intervention-policy).
  - **async**: pushes the record onto that binding's bounded queue and
    returns `None` immediately, without waiting for it to be processed. Can
    only ever use `intervention.mode=reject` — see below.
- **`end_request(request_id)`** — for each async binding, closes its queue
  and blocks until everything queued as of now has been drained and
  processed, *then* calls `on_end` on every probe for this request and
  tears down all of its state, returning
  `{extraction_point_name: ProbeResult}`. Works correctly even if
  generation was aborted mid-stream — whatever made it into the queue
  before `end_request` was called still gets folded in before the verdict
  is finalized.

## Intervention policy

`undercurrent.spec.InterventionPolicy` (`mode: reject | block_until_signal`,
`timeout_ms`, `on_timeout: continue | abort`) governs how long an inline
extraction point's `route()` call may block waiting for `on_activation`'s
decision. Set per extraction point (`ExtractionPoint.intervention`) or as
a router-wide fallback:

```python
from undercurrent.spec import InterventionPolicy, InterventionMode, TimeoutAction

router = Router(
    probe_registry,
    default_intervention_policy=InterventionPolicy(
        mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=250, on_timeout=TimeoutAction.ABORT
    ),
    circuit_breaker_threshold=5,  # see "Circuit breaker" below; this is the default
)
```

An extraction point's own `intervention` (if set) always wins over the
router's default; `None` (not specified) inherits the router's default. A
`block_until_signal` dispatch runs `on_activation` on the router's shared
executor and waits `future.result(timeout=timeout_ms / 1000)`; if it
doesn't finish in time (or raises), `route()` returns a synthetic
`ProbeSignal` instead: `action=abort` if `on_timeout=abort`, else
`action=continue`, with `metadata={"intervention_fallback": True, "reason":
"timeout"|"exception", ...}`. **The original call is not cancelled** — a
plain Python thread can't be preempted mid-call, so a straggling
`on_activation` keeps running in the background and its eventual result is
discarded; see `Router._dispatch_with_intervention`'s docstring for the
known race this implies for a probe with mutable state if a second call
gets dispatched before the first one actually finishes.

**`execution_mode=async` cannot use anything but the no-op default**
(`mode=reject`) — async is already fire-and-forget, so a `block_until_signal`
policy attached to one (directly, or inherited from a
`block_until_signal` router-wide default) has no defined semantics.
`undercurrent.spec`'s parser already rejects this at spec-parse time;
`register_request` re-checks it defensively for a hand-built
`ExtractionPoint` that bypassed the parser, exactly like the existing
`execution_mode=async` + `probe_kind=single_shot` check.

### Circuit breaker

If a `block_until_signal` extraction point racks up `circuit_breaker_threshold`
(default 5) **consecutive** timeouts/exceptions — counted by
`extraction_point_name`, deliberately across different `request_id`s, since
the whole point is catching a probe that's systematically unsafe, not
unlucky once — the router permanently downgrades that extraction point to
plain `reject` dispatch for every future `route()` call, for the rest of
this `Router`'s lifetime (a fresh success resets the counter to zero before
that point). Logs loudly via stdlib `logging`, and if a log sink is
attached (see below), forwards a synthetic `ProbeSignal(action=FLAG,
metadata={"circuit_breaker_tripped": True, ...})` through
`write_signal(...)` — the same "reuse the log-sink plumbing for an
out-of-band event" pattern `AsyncWorker` already uses for a probe's own
raised exceptions.

## Observation logging (`attach_log_sink`)

```python
router.attach_log_sink(sink)   # sink duck-types write_signal/write_result — see undercurrent.sinks
```

`Router.attach_log_sink(sink)` wires an out-of-band observation log onto
every **async** binding (i.e. `execution_mode=async`, which is only ever
valid with `probe_kind=trajectory` — "observe mode" probes): each
`ProbeSignal` such a probe emits from `on_activation` is forwarded to
`sink.write_signal(request_id, extraction_point_name, signal)` on that
binding's own worker thread, and its final `ProbeResult` is forwarded to
`sink.write_result(...)` once `end_request` finalizes it. Inline bindings
are unaffected — their signal already returns synchronously from `route()`.

`sink` isn't required to be an `undercurrent.sinks.LogSink`; it only needs to
duck-type `undercurrent.router.SupportsLogSink` (`write_signal`/`write_result`),
so this package takes no dependency on `undercurrent.sinks`.
[`undercurrent.sinks`](../../src/undercurrent/sinks/) is the package that implements
concrete sinks (`FileLogSink`, `WebhookLogSink`) against this hook.

Non-blocking guarantee: signal forwarding runs on the async binding's
dedicated worker thread, never inside `route()`'s caller, so a slow or
raising sink adds latency only to that one binding's own processing —
never to `route()`, and never to any other binding. A sink that raises is
logged and swallowed, the same way a misbehaving probe is. Call
`attach_log_sink` before or after `register_request`; it takes effect
immediately either way, and passing `None` detaches it.

## Isolation model

No structure in `Router` is ever keyed or aggregated across `request_id`s.
All per-request state — probe instances, async queues, the stored
`RequestContext` — lives nested under `request_id` in a single dict;
tearing down a `request_id` in `end_request` removes its entire subtree in
one step. Two concurrently registered requests that happen to reuse the
same `extraction_point_name` never see each other's probe state or
activation records (see `tests/test_isolation.py`, including a
multi-threaded variant driving 8 concurrent requests from separate
threads).

## Async dispatch mechanics

- **Bounded queue per binding** (`undercurrent.router.overflow.BoundedDropQueue`),
  sized by the extraction point's `queue_depth` (falling back to
  `Router`'s `default_queue_depth` when `queue_depth` is `None`, which
  `undercurrent.spec`'s schema allows even for async extraction points).
- **Worker pool**: a single shared `concurrent.futures.ThreadPoolExecutor`
  backs every async extraction point across every request. **Threads, not
  asyncio** — `Probe.on_activation` is a plain synchronous method with no
  guarantee it won't itself block (a real MLP forward pass, a GPU call),
  so running it inside an asyncio event loop would either stall the loop
  or require wrapping every call in `run_in_executor` anyway, which is a
  thread pool with extra steps. Threads also keep the router synchronous
  end-to-end, usable from ordinary adapter/test code with no event loop.
  Each async binding gets one dedicated long-lived drain-loop task on that
  pool for its whole lifetime — a single consumer, which is what
  guarantees activations are processed in the order `route()` enqueued
  them (required for a trajectory probe's state to be meaningful). The
  known tradeoff: if more concurrent async bindings are alive than
  `worker_pool_size`, the extras get no service until an earlier one's
  request ends and frees a thread — acceptable for this package's scope,
  called out explicitly in `Router`'s docstring.
- **Overflow policy** (`undercurrent.router.overflow.OverflowPolicy`):
  `drop_oldest` (default), `drop_newest`, or `block`. Read per extraction
  point via `getattr(point, "overflow_policy", None)` — `undercurrent.spec`'s
  schema doesn't define that field today, so this currently always falls
  through to `Router`'s `default_overflow_policy`, but the router honors a
  future schema addition without needing a code change here.

## Dependency shim

The stub-fallback dependency shim was removed in the single-package consolidation; shared types are imported directly.

## Response formatting (`response_schema`)

The OpenAI-compatible response formatting module that used to live here is
now a reference example, not package code: see
[`examples/openai_server/`](../../examples/openai_server/) and its tests in
`tests/examples/openai_server/`.

## Package layout

```
src/undercurrent/router/
  errors.py            RouterError
  overflow.py          OverflowPolicy, BoundedDropQueue
  binding.py           AsyncWorker (per-binding drain loop), Binding, LogSinkHolder, SupportsLogSink
  metrics.py           MetricsRecorder / MetricsSink / InMemoryMetricsRegistry
  router.py            Router
tests/router/          unit tests (pytest)
```

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest tests/router
```

# Embed in your serving stack

<!-- owner: p3-vllm-guide -->

`ProbedModel` owns the whole loop: it loads the model, runs generation and
hands you a `GenerationOutput`. This page is for when you need to own that
loop yourself. It shows how to drive an engine adapter and the `Router`
directly, and which settings to tune for throughput and latency.

## When to go below `ProbedModel`

Stay with `ProbedModel` (see [Deploy with vLLM](../guides/vllm-deployment.md))
unless one of these applies:

- **You own the request loop.** Requests arrive from your queue, scheduler or
  batch job, and you decide when each one starts, which spec it gets, and what
  happens to its results.
- **You have your own server.** You already run FastAPI, gRPC or an internal
  RPC framework, and you want probing inside its request handlers, with your
  own request ids, timeouts and response format.
- **You need custom routing.** Different requests get different extraction
  points, or several engines share one `Router`, or activations come from an
  engine Undercurrent has no adapter for, so you build the `ActivationRecord`s
  yourself.

`ProbedModel` is built from the same pieces, so you can also go part of the
way: `ProbedModel(router=...)` takes a `Router` you configured, and
`ProbedModel(backend=<adapter>)` takes an adapter you loaded.

## The lifecycle

The `Router` is long-lived: build one at startup and share it across every
request and thread. Each request goes through the same steps:

| Step | Call | Who calls it |
| --- | --- | --- |
| 1. Start the request: spawn a fresh probe instance per extraction point, call `on_start` | `router.register_request(request_id, extraction_points, request_ctx)` | The engine adapter, inside `generate()`; or you, for your own engine |
| 2. One call per matching activation; returns the inline probe's `ProbeSignal` (or `None`) | `router.route(record)` | The engine adapter; or you |
| 3. Drain async queues, call `on_end`, return `{extraction_point_name: ProbeResult}` | `router.end_request(request_id)` | The engine adapter, when generation finishes or aborts; or you |
| 4. At process exit: drain or cancel what's left, stop the worker pool | `router.shutdown(wait=True)` | You |

`with router.request(...) as req:` wraps steps 1 to 3 and always ends the
request, even if your code raises. `with Router(...) as router:` calls
`shutdown(wait=True)` when the block exits.

When you drive an engine adapter, the adapter makes the per-request router
calls for you. Your side of the contract is:

1. `adapter.load_model(model, **engine_kwargs)` once at startup.
2. Per request: `adapter.register_extraction(request_id, points)`, then
   `text = adapter.generate(request_id, prompt, generation_kwargs, router)`,
   then `adapter.unregister_extraction(request_id)` in a `finally` block.
3. `router.on_request_end(listener)` to receive each request's results,
   because `generate()` returns only the text. The listener is called as
   `listener(request_id, results)`, from the thread that ended the request.

### Your own engine: synthetic records

This example plays the part of an engine. It uses no model, so it runs
anywhere. There are two extraction points: an inline gate that can stop
generation, and an async trajectory probe that only observes. The router is
configured the way you would for production, with an explicit pool size,
queue depth, overflow policy, drain timeout, metrics sink and log sink.

```python
import json
import tempfile
import threading
from collections import Counter
from pathlib import Path

from undercurrent.core import ActivationRecord, Probe, ProbeAction, ProbeResult, ProbeSignal, probe
from undercurrent.router import MetricsSink, OverflowPolicy, Router
from undercurrent.sinks import FileLogSink
from undercurrent.spec import parse_yaml

SPEC = parse_yaml("""
version: "1"
extraction_points:
  - name: spike_gate              # inline: can stop generation
    layers: 6
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: spike
    probe_kind: single_shot
    execution_mode: inline
  - name: drift_watch             # async: observes, never blocks
    layers: 6
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: running_mean
    probe_kind: trajectory
    execution_mode: async
    queue_depth: 64
""")


@probe("spike", threshold=8.0, register=False)
def spike(record: ActivationRecord) -> float:
    return max(record.tensor)


class RunningMean(Probe):
    probe_kind = "trajectory"

    def on_start(self, request_ctx):
        self.values = []

    def on_activation(self, record):
        self.values.append(sum(record.tensor) / len(record.tensor))
        if self.values[-1] > 3.0:
            return ProbeSignal(action=ProbeAction.FLAG, confidence=self.values[-1])
        return None

    def on_end(self, request_ctx):
        mean = sum(self.values) / len(self.values) if self.values else None
        return ProbeResult(
            request_ctx.request_id, self.extraction_point_name, verdict={"mean": mean, "n": len(self.values)}
        )


class CountingMetrics(MetricsSink):
    """Totals per extraction point. Thread-safe, no I/O: a real sink would
    update Prometheus or OpenTelemetry instruments here."""

    def __init__(self):
        self._lock = threading.Lock()
        self.totals = Counter()

    def record_queue_depth(self, request_id, extraction_point_name, depth):
        pass  # a gauge in a real backend

    def record_drop(self, request_id, extraction_point_name):
        with self._lock:
            self.totals[extraction_point_name, "drops"] += 1

    def record_activation(self, request_id, extraction_point_name, latency_seconds):
        with self._lock:
            self.totals[extraction_point_name, "activations"] += 1

    def record_probe_error(self, request_id, extraction_point_name):
        with self._lock:
            self.totals[extraction_point_name, "errors"] += 1


def fake_engine(request_id, prompt_len, values):
    """Stand-in for your engine: one activation per generated token, for
    every extraction point whose position selector matches that token."""
    for i, value in enumerate(values):
        token_index = prompt_len + i
        for point in SPEC:
            if point.matches(token_index, is_generated=True, prompt_len=prompt_len, generated_index=i):
                yield ActivationRecord(
                    request_id=request_id,
                    extraction_point_name=point.name,
                    layer=6,
                    token_pos=token_index,
                    tensor_type="residual_stream",
                    tensor=[value, value / 2, value / 4],
                    is_generated=True,
                )


metrics = CountingMetrics()
log_path = Path(tempfile.mkdtemp()) / "observations.ndjson"

with Router(
    {"spike": spike, "running_mean": RunningMean},
    worker_pool_size=8,
    default_queue_depth=32,
    default_overflow_policy=OverflowPolicy.DROP_OLDEST,
    drain_timeout=5.0,
    metrics_sink=metrics,
) as router:
    router.attach_log_sink(FileLogSink(log_path))

    for request_id, values in [("req-calm", [1.0, 2.0, 3.0, 2.0]), ("req-spiky", [2.0, 6.0, 9.0, 12.0])]:
        with router.request(SPEC, request_id=request_id, prompt_metadata={"prompt_len": 5}) as req:
            for record in fake_engine(request_id, prompt_len=5, values=values):
                signal = req.route(record)  # None for async points
                if signal is not None and signal.action is ProbeAction.ABORT:
                    print(f"{request_id}: abort at token {record.token_pos}")
                    break  # your engine stops generating here
        print(request_id, {name: result.verdict for name, result in req.results.items()})

    assert req.results["spike_gate"].verdict["flagged"] is True

print(dict(metrics.totals))
for line in log_path.read_text().splitlines():
    entry = json.loads(line)
    print(entry["kind"], entry["request_id"], entry["extraction_point_name"])
```

The calm request runs to the end. The spiky one is stopped by `spike_gate` at
its third generated token, and `drift_watch` still returns a result covering
the activations it received before the abort. The metrics sink counts the
async point's activations, and the log file holds the async probe's `flag`
signals and one `result` line per request. Inline points never reach the log
sink or the metrics sink: their signal goes straight back to the caller of
`route()`.

### Driving the Hugging Face adapter

With a real model, the adapter produces the records and makes the router
calls. This runs GPT-2 on the CPU (downloaded from the Hugging Face Hub the
first time). The HF adapter runs one `generate()` at a time.

```python
from undercurrent.adapters.hf import HFEngineAdapter

GATE_SPEC = parse_yaml("""
version: "1"
extraction_points:
  - name: last_prompt_token
    layers: 6
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: norm
    probe_kind: single_shot
    execution_mode: inline
""")


@probe("norm", threshold=500.0, register=False)
def norm(record: ActivationRecord) -> float:
    return float(record.tensor.norm())


results_by_request = {}
results_lock = threading.Lock()


def collect(request_id, results):
    with results_lock:
        results_by_request[request_id] = results


adapter = HFEngineAdapter()
adapter.load_model("gpt2", device="cpu")

with Router({"norm": norm}) as router:
    remove_listener = router.on_request_end(collect)
    for request_id, prompt in [("req-1", "The quick brown fox"), ("req-2", "Once upon a time")]:
        adapter.register_extraction(request_id, list(GATE_SPEC))
        try:
            text = adapter.generate(
                request_id,
                prompt,
                {"max_new_tokens": 10, "do_sample": False, "pad_token_id": adapter.tokenizer.eos_token_id},
                router,
            )
        finally:
            adapter.unregister_extraction(request_id)
        print(request_id, repr(text), results_by_request[request_id]["last_prompt_token"].verdict)
    remove_listener()

adapter.close()
assert set(results_by_request) == {"req-1", "req-2"}
```

`generation_kwargs` go to the engine as they are: here to `model.generate()`,
for vLLM to `SamplingParams`. `ProbedModel`'s normalised names
(`max_new_tokens`, `temperature`, `stop`, ...) are not translated at this
level.

### The vLLM equivalent

!!! info "Requires a GPU"
    This example needs a CUDA GPU and vLLM.

`VLLMEngineAdapter.generate()` is safe to call from many threads at once:
each call blocks only its own thread, and vLLM's scheduler batches the
requests together. Use one adapter and one `Router` per process, and call
`generate()` from your server's request threads (or a thread pool):

```py
import uuid
from concurrent.futures import ThreadPoolExecutor

from undercurrent.adapters.vllm import VLLMEngineAdapter
from undercurrent.router import Router

router = Router({"norm": norm}, worker_pool_size=64)
router.on_request_end(collect)

adapter = VLLMEngineAdapter()  # checks the installed vLLM version
adapter.load_model("gpt2", gpu_memory_utilization=0.3, max_model_len=1024, enforce_eager=True)


def handle(prompt: str) -> dict:
    request_id = uuid.uuid4().hex
    adapter.register_extraction(request_id, list(GATE_SPEC))
    try:
        text = adapter.generate(request_id, prompt, {"max_tokens": 64, "temperature": 0.0}, router)
    finally:
        adapter.unregister_extraction(request_id)
    with results_lock:
        results = results_by_request.pop(request_id, {})
    return {"text": text, "verdicts": {name: r.verdict for name, r in results.items()}}


try:
    with ThreadPoolExecutor(max_workers=32) as pool:
        for response in pool.map(handle, ["Hello, my name is", "The capital of France is"]):
            print(response)
finally:
    adapter.shutdown()  # stops the adapter's background event loop
    router.shutdown(wait=True)
```

Pop each request's results from your listener's store as you use them, as
`handle()` does, so the store doesn't grow for the life of the process.
[`examples/openai_server/`](https://github.com/wrynx/undercurrent/tree/main/examples/openai_server)
is this same pattern inside an HTTP server.

## Tune for production

Each knob below has a page with the details; this section is the summary.
All `Router` settings can also reach a `ProbedModel` through
`router_kwargs={...}`.

### Inline or async, per extraction point

`execution_mode` is set per extraction point. An **inline** probe runs in
`route()`, on the generation path: it can stop generation, and every
activation it handles adds its run time to that token's latency. An **async**
probe gets a queue and a worker thread: `route()` returns at once, and the
probe can only observe. Make a point inline only if its decision has to stop
the generation, and keep that probe cheap. Make everything else async.
On vLLM, an inline probe's run time is paid by every request in the batch.
See [Execution modes](../concepts/execution-modes.md).

### Queue depth and overflow policy

Each async binding has a bounded queue: `queue_depth` per extraction point in
the spec, or `Router(default_queue_depth=...)` (default 32) for points that
don't set one. When a queue is full, the router-wide
`Router(default_overflow_policy=...)` decides:

| Policy | Generation latency | What you lose |
| --- | --- | --- |
| `OverflowPolicy.DROP_OLDEST` (default) | Unaffected | The oldest queued activations; the probe sees the most recent ones |
| `OverflowPolicy.DROP_NEWEST` | Unaffected | The incoming activations; the probe sees the earliest ones |
| `OverflowPolicy.BLOCK` | `route()` waits for space, so generation slows to the probe's pace (on vLLM, the whole batch does) | Nothing |

Use a drop policy for live serving and watch the drop count; use `BLOCK` for
offline evaluation where every activation must be processed. A bigger queue
absorbs bursts (prefill delivers many activations at once) but doesn't help a
probe that is slower than generation on average. See
[Async execution & backpressure](async-execution.md#sizing).

### Worker counts

Every live async binding holds one thread from the router's shared pool for
its request's whole lifetime. Bindings that can't get a thread don't run:
their queues fill up and apply the overflow policy. Size `worker_pool_size`
from concurrency, not CPU cores:

```text
worker_pool_size >= max concurrent requests × async extraction points per request
                    + headroom for block_until_signal calls
```

The default is `min(32, 4 × CPU count)`. With vLLM, the number of concurrent
requests is your server's concurrency (`ProbedModel`'s `max_concurrency`
defaults to 64), so the default pool is usually too small once you have async
points. See
[Async execution & backpressure](async-execution.md#worker_pool_size).

### Intervention timeouts

By default an inline probe runs for as long as it takes (`mode: reject`). To
bound it, give the extraction point an `intervention` with
`mode: block_until_signal`, a `timeout_ms`, and `on_timeout: abort` (fail
closed) or `continue` (fail open). After `circuit_breaker_threshold`
consecutive timeouts or errors (default 5), the router downgrades that
extraction point to `reject` for the rest of the process and logs a warning.
On vLLM, the bounded wait holds the whole batch, so keep `reject` under
concurrent traffic. See
[Intervention policies & timeouts](intervention-policies.md#recommended-production-settings).

### Metrics

The router always keeps per-binding counters (`router.get_metrics(request_id,
extraction_point_name)`, while the request is live). To export them, pass a
`MetricsSink` as `Router(metrics_sink=...)` or call
`router.attach_metrics_sink(sink)`, as `CountingMetrics` does above. Your
sink is called from the generation path and from worker threads, so it must
be thread-safe and must not block: update in-memory instruments and let your
exporter ship them. Label by `extraction_point_name`, never by `request_id`.
Alert on drops above zero and on queue depth sitting at its maximum. See
[Metrics](metrics.md).

### Log sink

`router.attach_log_sink(sink)` sends every async probe's signals and final
results to a sink: `FileLogSink` for NDJSON on local disk, `WebhookLogSink`
for HTTP delivery with retries, dead-lettering and redaction. Only async
extraction points are logged. See
[Observation sinks](../guides/observation-sinks.md).

## Operational checklist

**Graceful shutdown.** Stop accepting new requests and let in-flight ones
finish. Then shut down in this order:

1. the engine adapter (`VLLMEngineAdapter.shutdown()`, or
   `HFEngineAdapter.close()`);
2. `router.shutdown(wait=True)`, or leave the `with Router(...)` block;
3. your sinks, for example `WebhookLogSink.close(timeout=...)`, so the
   router's last results reach them first.

**Per-request isolation.** Every request gets its own probe instance per
extraction point, its own async queues and its own metrics entry, all keyed by
`request_id` and torn down by `end_request`. Concurrent requests never share
probe state. Two things are shared on purpose: the worker pool (sized as
above) and the intervention circuit breaker, which counts failures per
extraction point name across requests. Request ids must be unique among
in-flight requests; registering one that is already live raises
`RouterError`.

**In-flight async work.** At the end of a request, `end_request` processes
everything already queued (up to `drain_timeout` per binding, default 30
seconds) before calling `on_end`, so an async verdict covers every activation
that wasn't dropped. If a binding doesn't drain in time, the router logs a
warning and finalizes it anyway. At process exit:

- `router.shutdown(wait=True)` drains every still-registered request's queues,
  then stops the pool.
- `router.shutdown(wait=False)` discards queued work and returns at once. A
  probe call already running still finishes.

Either way, requests still registered at shutdown never get `on_end`, so they
produce no result and nothing reaches your `on_request_end` listener or log
sink for them. Let requests end normally before you shut down.

**Where errors surface.**

| What failed | Where you see it |
| --- | --- |
| An invalid spec or unknown `probe_type` | `RouterError` from `register_request` (or `router.request(...)`); nothing is registered for that request. `ProbedModel` raises `ProbedModelConfigError` at construction instead. |
| An inline probe raises (`mode: reject`) | The exception propagates out of `route()`. With the HF adapter, and on the vLLM adapter's polling path, it comes out of `adapter.generate()`, which still calls `end_request`. On the vLLM adapter's in-process path it is raised inside the engine's forward pass, where it can affect the other requests in the batch. Make inline probes exception-safe. |
| An inline probe raises or times out (`mode: block_until_signal`) | `route()` returns the `on_timeout` fallback signal, with `metadata["intervention_fallback"] = True` and the reason. Logged on the `undercurrent.router` logger; counts towards the circuit breaker. |
| An async probe raises | Logged on the `undercurrent.router` logger, counted by `record_probe_error`, and forwarded to the log sink as a synthetic signal with `metadata["router_error"] = True`. The binding keeps processing. |
| A metrics sink, log sink or `on_request_end` listener raises | Logged and ignored; dispatch and other listeners are unaffected. |
| `route()` for a request that has ended, or after `shutdown` | `RouterError`. |
| vLLM version, topology or internals mismatch | Errors from the vLLM adapter; see [Troubleshooting](../guides/vllm-deployment.md#troubleshooting). |

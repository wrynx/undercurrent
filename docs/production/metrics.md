# Metrics

<!-- owner: p3-sinks-guide -->

The router measures every **async** binding: one probe instance for one
extraction point in one request. For each binding it records queue depth,
overflow drops, `on_activation` latency and probe errors. You can read the
numbers in two ways:

- **Pull:** call `router.get_metrics(request_id, extraction_point_name)` for a
  point-in-time `MetricsSnapshot`. This always works and needs no setup.
- **Push:** attach your own `MetricsSink` with `router.attach_metrics_sink(sink)`
  and forward each event to Prometheus, OpenTelemetry, StatsD or any other
  backend.

Inline extraction points aren't measured. Their probe runs synchronously in
`route()`, so you can time it yourself at the call site. All the names on
this page come from `undercurrent.router.metrics` and `undercurrent.router`.

## The events: `MetricsSink`

`MetricsSink` is an abstract base class with four methods. Every call carries
the binding's `request_id` and `extraction_point_name`.

| Method | When it's called | What it means |
| --- | --- | --- |
| `record_queue_depth(request_id, extraction_point_name, depth)` | Right after each change to the queue: after `route()` enqueues an activation, and after the worker takes one off. | A gauge: how many activations are waiting for this probe. |
| `record_drop(request_id, extraction_point_name)` | Once for every activation that the queue's overflow policy discards. Under `drop_oldest` that is the evicted oldest item; under `drop_newest` it's the rejected new item. It's **not** called for a `put` that fails because the queue was already closed during shutdown. | A counter: activations the probe never saw. |
| `record_activation(request_id, extraction_point_name, latency_seconds)` | Once for every `on_activation` call the worker made, **whether it returned or raised**. | Wall-clock seconds spent inside `on_activation`. Sink forwarding and queue wait aren't included. |
| `record_probe_error(request_id, extraction_point_name)` | Once for every `on_activation` call that raised. | A counter: probe failures. Each one also sends a synthetic error signal to the [log sink](../guides/observation-sinks.md). |

So the error rate of a binding is `probe errors / activations`, and the
drop rate is `drops / (activations + drops + queue depth)`, roughly the
activations that were routed.

### Which thread calls what

Your sink is called from two places:

- `record_activation` and `record_probe_error`, plus the queue-depth reading
  after each dequeue, come from the binding's **own worker thread**.
- The queue-depth reading after an enqueue, and `record_drop`, come from the
  **thread that called `route()`**, which is the engine's generation path.
  `record_drop` is even called while the binding's queue lock is held.

Your `MetricsSink` must therefore be:

- **Thread-safe.** Many bindings call it concurrently.
- **Fast and non-blocking.** Increment a counter or set a gauge, and do no
  I/O. If you need to send data over the network, do it from a separate
  exporter thread, the way Prometheus' pull model and OpenTelemetry's
  periodic readers already do.
- **Free of calls back into the router.**

If your sink raises, the router catches the exception, logs it with
`logger.exception`, and carries on. Dispatch and probe processing are never
affected, but the event is lost for your sink. The built-in registry still
counts it.

## The built-in registry and `get_metrics`

Every `Router` builds an `InMemoryMetricsRegistry` and always records into
it, whether or not you attach a sink. It can't be replaced; it exists so
that `get_metrics` works with no setup.

`router.get_metrics(request_id, extraction_point_name)` returns a frozen
`MetricsSnapshot`:

| Field | Type | Meaning |
| --- | --- | --- |
| `queue_depth` | `int` | The last value from `record_queue_depth`. |
| `drop_count` | `int` | The number of `record_drop` calls. |
| `activation_count` | `int` | The number of `record_activation` calls (completed `on_activation` calls). |
| `error_count` | `int` | The number of `record_probe_error` calls. |
| `avg_activation_latency_seconds` | `float` or `None` | The mean of the `latency_seconds` values, or `None` before the first activation. |

The registry only keeps state per request:

- Each binding's entry is created zeroed by `register_request` and **deleted
  by `end_request`** (and by `shutdown`).
- `get_metrics` raises `RouterError` for an unknown or already-ended
  `request_id`, an unknown extraction point name, or an inline extraction
  point.
- Nothing is aggregated across requests.

If you need totals across requests or over time, which is what a dashboard
needs, attach a `MetricsSink` and aggregate there.

### Example: watch a slow probe drop activations

The probe below blocks on its first activation until the example releases
it. While it's blocked, the example routes more activations than its queue
can hold, so the `drop_oldest` overflow policy has to evict some of them. A
tiny counting `MetricsSink` keeps totals that outlive the request.

```python
import collections
import threading

from undercurrent.core import Probe, ProbeFactory, ProbeResult
from undercurrent.router import MetricsSink, Router
from undercurrent.spec import ActivationRecord, parse_yaml

SPEC = parse_yaml("""
version: "1"
extraction_points:
  - name: slow_probe
    layers: [4]
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: slow
    probe_kind: trajectory
    execution_mode: async
    queue_depth: 4
""")

started = threading.Event()
release = threading.Event()


class SlowProbe(Probe):
    """Blocks on its first activation until released; raises on negative inputs."""

    probe_kind = "trajectory"

    def on_start(self, request_ctx):
        self.seen = 0

    def on_activation(self, record):
        started.set()
        release.wait(timeout=10)
        if record.tensor[0] < 0:
            raise ValueError("negative activation")
        self.seen += 1
        return None

    def on_end(self, request_ctx):
        return ProbeResult(self.request_id, self.extraction_point_name, verdict={"seen": self.seen})


class CountingMetricsSink(MetricsSink):
    """Totals per extraction point, across all requests. Thread-safe and O(1) per event."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counters = collections.Counter()
        self.max_queue_depth = collections.defaultdict(int)

    def record_queue_depth(self, request_id, extraction_point_name, depth):
        with self._lock:
            self.max_queue_depth[extraction_point_name] = max(self.max_queue_depth[extraction_point_name], depth)

    def record_drop(self, request_id, extraction_point_name):
        with self._lock:
            self.counters[(extraction_point_name, "drops")] += 1

    def record_activation(self, request_id, extraction_point_name, latency_seconds):
        with self._lock:
            self.counters[(extraction_point_name, "activations")] += 1

    def record_probe_error(self, request_id, extraction_point_name):
        with self._lock:
            self.counters[(extraction_point_name, "errors")] += 1


def activation(request_id, pos, value):
    return ActivationRecord(
        request_id=request_id,
        extraction_point_name="slow_probe",
        layer=4,
        token_pos=pos,
        tensor_type="residual_stream",
        tensor=[value],
        is_generated=True,
    )


metrics = CountingMetricsSink()
with Router(probe_registry={"slow": ProbeFactory(SlowProbe)}) as router:
    router.attach_metrics_sink(metrics)  # or Router(..., metrics_sink=metrics)

    with router.request(SPEC, request_id="req-1") as req:
        req.route(activation("req-1", 0, 1.0))
        started.wait(timeout=10)  # the worker is now stuck inside on_activation

        # 9 more activations into a queue of 4: the 5 oldest are evicted.
        for pos in range(1, 10):
            req.route(activation("req-1", pos, -1.0 if pos == 9 else 1.0))

        snap = router.get_metrics("req-1", "slow_probe")
        print(snap)
        assert (snap.queue_depth, snap.drop_count, snap.activation_count) == (4, 5, 0)
        assert snap.avg_activation_latency_seconds is None  # nothing has finished yet

        release.set()  # let the probe catch up; leaving the block drains the queue

# The per-request entry is gone now; the external sink kept the totals.
print(dict(metrics.counters), dict(metrics.max_queue_depth))
assert metrics.counters[("slow_probe", "drops")] == 5
assert metrics.counters[("slow_probe", "activations")] == 5  # the first one plus the 4 that survived
assert metrics.counters[("slow_probe", "errors")] == 1  # the last activation was negative
assert metrics.max_queue_depth["slow_probe"] == 4
assert req.results["slow_probe"].verdict == {"seen": 4}
```

The code prints the following, plus a logged traceback for the deliberate
`ValueError`:

```console
MetricsSnapshot(queue_depth=4, drop_count=5, activation_count=0, error_count=0, avg_activation_latency_seconds=None)
{('slow_probe', 'drops'): 5, ('slow_probe', 'activations'): 5, ('slow_probe', 'errors'): 1} {'slow_probe': 4}
```

`get_metrics` is useful for tests, debugging endpoints and adaptive logic
inside a request. For anything longer-lived, use a `MetricsSink`.

## Sketch: export to Prometheus

!!! warning "Sketch, not shipped"
    Undercurrent doesn't include a Prometheus or OpenTelemetry exporter, and
    doesn't depend on `prometheus_client`. The code below is an illustrative
    starting point. Adapt and test it in your own stack.

**Don't use `request_id` as a label.** Every request creates new bindings,
and a per-request label set would grow without bound. Label by
`extraction_point_name` only, since it comes from your spec and has a fixed
set of values.

```py
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from undercurrent.router import MetricsSink

ACTIVATIONS = Counter("undercurrent_probe_activations_total", "on_activation calls completed", ["extraction_point"])
ERRORS = Counter("undercurrent_probe_errors_total", "on_activation calls that raised", ["extraction_point"])
DROPS = Counter("undercurrent_queue_drops_total", "Activations discarded by the overflow policy", ["extraction_point"])
LATENCY = Histogram(
    "undercurrent_probe_activation_seconds",
    "Wall-clock time inside on_activation",
    ["extraction_point"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0),
)
QUEUE_DEPTH = Gauge(
    "undercurrent_queue_depth",
    "Most recent queue depth reported by any binding of this extraction point",
    ["extraction_point"],
)


class PrometheusMetricsSink(MetricsSink):
    # prometheus_client metrics are thread-safe and in-memory, so every method is cheap.

    def record_queue_depth(self, request_id, extraction_point_name, depth):
        QUEUE_DEPTH.labels(extraction_point_name).set(depth)

    def record_drop(self, request_id, extraction_point_name):
        DROPS.labels(extraction_point_name).inc()

    def record_activation(self, request_id, extraction_point_name, latency_seconds):
        ACTIVATIONS.labels(extraction_point_name).inc()
        LATENCY.labels(extraction_point_name).observe(latency_seconds)

    def record_probe_error(self, request_id, extraction_point_name):
        ERRORS.labels(extraction_point_name).inc()


start_http_server(9400)  # /metrics on its own thread
router.attach_metrics_sink(PrometheusMetricsSink())
```

`QUEUE_DEPTH` is "last write wins" across all concurrent requests for that
extraction point. It's a rough signal, good enough to spot a backlog. For
an exact per-binding maximum, keep a dict keyed by
`(request_id, extraction_point_name)` in the sink, remove keys you haven't
seen for a while, and export the maximum as the gauge.

With OpenTelemetry the shape is the same: create `Counter`s and a
`Histogram` from a `Meter`, call `.add(1, {"extraction_point": name})` and
`.record(latency, {...})` in the four methods, and let the SDK's periodic
reader export them. For queue depth, use an `UpDownCounter`, or an
observable gauge that reads a dict your sink maintains.

## Suggested alerts

Use these as starting points and tune the thresholds to your traffic. For
how queue depth, overflow policies and the worker pool interact, see
[Async execution & backpressure](async-execution.md).

| Alert | Example PromQL (with the sketch above) | What it usually means |
| --- | --- | --- |
| **Sustained drops** | `sum by (extraction_point) (rate(undercurrent_queue_drops_total[5m])) > 0` for 10m | The probe can't keep up with the activation rate. Its results are computed from a subsample. Speed up the probe, add `stride` to the spec, raise `queue_depth`, or raise `worker_pool_size` if the bindings are waiting for a worker. |
| **Drop ratio** | `rate(undercurrent_queue_drops_total[5m]) / (rate(undercurrent_queue_drops_total[5m]) + rate(undercurrent_probe_activations_total[5m])) > 0.01` | Same as above, normalized for traffic. Use it to page if verdict quality depends on seeing every activation. |
| **Queue depth near the limit** | `max by (extraction_point) (undercurrent_queue_depth) >= 0.8 * <queue_depth from your spec>` for 5m | Drops are imminent. Usually a slow probe or a slow custom log sink. Activation latency tells you which: if it's flat while depth climbs, look at the sink. |
| **Probe error rate** | `rate(undercurrent_probe_errors_total[5m]) / rate(undercurrent_probe_activations_total[5m]) > 0.01` | The probe is raising. The error signals in your [observation sink](../guides/observation-sinks.md) carry `error_type` and the message. |
| **Activation latency** | `histogram_quantile(0.99, sum by (le, extraction_point) (rate(undercurrent_probe_activation_seconds_bucket[5m])))` above your budget | The probe is getting slower. Queue depth and drops follow if the latency doesn't fit between tokens. |

The router doesn't emit metrics for inline extraction points, intervention
timeouts or circuit-breaker trips. Timeouts and trips are logged instead.
See [Intervention policies & timeouts](intervention-policies.md).

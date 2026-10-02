# Execution modes: inline vs async

<!-- owner: p3-concepts -->

Every extraction point runs in one of two **execution modes**, set by
`execution_mode` in the spec:

- **`inline`** (the default): the probe runs synchronously, on the generation
  path, for each matching activation. Its `ProbeSignal` goes straight back to
  the engine adapter, so an inline probe can **stop generation**. The cost is
  that generation waits for it.
- **`async`**: the activation is put on a queue and the probe runs on a
  background worker thread. Generation never waits, but the probe can't stop
  it: it **observes**, and its signals and result go to a
  [log sink](../guides/observation-sinks.md).

| | `inline` | `async` |
| --- | --- | --- |
| Where the probe runs | The adapter's thread, inside `router.route()` | A worker thread from the router's pool |
| `route()` returns | The probe's `ProbeSignal` (or `None`) | Always `None`, immediately |
| Can abort generation | Yes | No |
| Adds latency to generation | Yes, the probe's run time per matching activation | No (unless the `block` overflow policy is chosen) |
| Where signals go | Back to the adapter | To the attached log sink |
| Allowed `probe_kind` | `single_shot` or `trajectory` | `trajectory` only |
| Allowed `intervention.mode` | `reject` or `block_until_signal` | `reject` only |
| Extra settings | [`intervention`](interventions.md) | `queue_depth`, overflow policy |

Choose **inline** when the probe's job is to act: block unsafe output, stop a
runaway generation. Keep inline probes fast. Choose **async** when the job is
to watch and record: trajectory analysis, monitoring, collecting data for
offline review.

## Example

The spec below runs the same running-mean probe twice: once inline (it can
abort) and once async (it only logs).

```yaml
extraction_points:
  - name: guard
    layers: 4
    tensor_type: mlp_out
    position: "generated[*]"
    probe_type: running_mean
    probe_kind: trajectory
    execution_mode: inline
  - name: monitor
    layers: 4
    tensor_type: mlp_out
    position: "generated[*]"
    probe_type: running_mean
    probe_kind: trajectory
    execution_mode: async
    queue_depth: 16
```

Here is the same setup driven with synthetic activations. The log sink is
just an object with `write_signal` and `write_result` methods; in production
you'd use `FileLogSink` or `WebhookLogSink` from `undercurrent.sinks`.

```python
from undercurrent.core import ProbeAction, ProbeFactory, RequestContext
from undercurrent.core.examples import TrajectoryScoreProbe
from undercurrent.router import Router
from undercurrent.spec import ActivationRecord, ExecutionMode, ExtractionPoint, ProbeKind, TensorType, parse_position


def make_point(name, mode, queue_depth=None):
    return ExtractionPoint(
        name=name,
        layers=(4,),
        tensor_type=TensorType.MLP_OUT,
        position=parse_position("generated[*]"),
        stride=None,
        until=None,
        probe_type="running_mean",
        probe_kind=ProbeKind.TRAJECTORY,
        execution_mode=mode,
        queue_depth=queue_depth,
    )


class ListSink:
    """Collects everything async probes report."""

    def __init__(self):
        self.signals, self.results = [], []

    def write_signal(self, request_id, extraction_point_name, signal):
        self.signals.append((extraction_point_name, signal.action))

    def write_result(self, request_id, extraction_point_name, result):
        self.results.append((extraction_point_name, result.verdict))


points = [make_point("guard", ExecutionMode.INLINE), make_point("monitor", ExecutionMode.ASYNC, queue_depth=16)]
router = Router(probe_registry={"running_mean": ProbeFactory(TrajectoryScoreProbe, {"threshold": 0.5})})
sink = ListSink()
router.attach_log_sink(sink)
router.register_request("req-1", points, RequestContext("req-1", {}, None))

stopped_at = None
for step, values in enumerate([[0.1, 0.2], [0.4, 0.6], [0.9, 1.0], [0.9, 0.9]]):
    for point in points:
        record = ActivationRecord("req-1", point.name, 4, 10 + step, "mlp_out", values, True)
        signal = router.route(record)
        if point.name == "monitor":
            assert signal is None  # async: never a signal back to the caller
        elif signal is not None and signal.action is ProbeAction.ABORT:
            stopped_at = step  # inline: an adapter would stop generating now
    if stopped_at is not None:
        break

results = router.end_request("req-1")  # drains the async queue first
print("aborted at step", stopped_at)  # aborted at step 2
print(results["monitor"].verdict)  # the async probe saw the same 3 activations
print(sink.signals)  # [('monitor', <ProbeAction.ABORT: 'abort'>)] -- logged, not acted on
router.shutdown()
```

The async probe reached the same conclusion, but its `abort` only went to the
sink: an async probe can't stop generation.

## How async dispatch works

Each async extraction point of each request gets its own **binding**:

- **A bounded queue.** `queue_depth` (from the spec, or the router's
  `default_queue_depth`, 32) caps how many activations can wait. Each request
  gets its own queue, so one slow request can't fill another's.
- **One worker.** A single long-lived task on the router's shared thread pool
  drains that queue. One consumer per binding means the probe sees its
  activations in exactly the order they were routed, which trajectory probes
  depend on.
- **An overflow policy** decides what happens when the queue is full:

| Policy | When the queue is full | Trade-off |
| --- | --- | --- |
| `drop_oldest` (default) | Evict the oldest waiting activation, accept the new one | Generation never waits; the probe loses older activations |
| `drop_newest` | Discard the incoming activation | Generation never waits; the probe loses the latest activations |
| `block` | `route()` waits until there is room | Nothing is dropped; generation slows down to the probe's pace |

Every drop is counted in the router's [metrics](../production/metrics.md), so
you can see when a probe falls behind.

When a request ends, `end_request` closes each async queue, waits for the
worker to process everything still queued (up to the router's
`drain_timeout`, 30 s by default), and only then calls `on_end`. That means an
async probe's verdict includes every activation that made it into its queue,
even if generation was aborted part-way.

Operating this at scale (sizing queues and the worker pool, choosing an
overflow policy, reading backpressure metrics, shutdown behaviour) is covered
in [Async execution & backpressure](../production/async-execution.md).

## Observation logging

`router.attach_log_sink(sink)` connects a sink to every **async** binding:

- every non-`None` `ProbeSignal` an async probe returns is passed to
  `sink.write_signal(request_id, extraction_point_name, signal)`, from that
  binding's worker thread;
- each async probe's final `ProbeResult` is passed to
  `sink.write_result(request_id, extraction_point_name, result)` when
  `end_request` finalizes it.

Inline extraction points don't forward to the sink: their signal already goes
back to the caller of `route()`. A sink only needs those two methods (it
doesn't have to subclass `undercurrent.sinks.LogSink`). A slow or failing
sink never slows down `route()`: forwarding happens on the worker thread, and
exceptions from the sink are logged and swallowed. You can attach a sink
before or after registering requests, and `attach_log_sink(None)` detaches it.

The built-in sinks:

```py
from undercurrent.sinks import FileLogSink, WebhookLogSink

router.attach_log_sink(FileLogSink("observations.ndjson"))

# or: POST each record, retry with backoff, dead-letter to a local file on failure
router.attach_log_sink(WebhookLogSink("https://example.com/hook", dead_letter_path="webhook_dead_letters.ndjson"))
```

Both write one JSON record per signal or result, with tensors summarized as
shape and dtype rather than dumped. See
[Observation sinks](../guides/observation-sinks.md) for the record format,
retries and redaction.

## Next

- [Interventions](interventions.md): what an inline probe can do to generation.
- [`undercurrent.router` API reference](../reference/router.md),
  [`undercurrent.sinks` API reference](../reference/sinks.md).

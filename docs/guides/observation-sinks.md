# Observation sinks

<!-- owner: p3-sinks-guide -->

!!! tip "Just want to see results?"
    If all you need is to look at probe results in the same Python process,
    use the simpler hook on the front-door API instead:
    `ProbedModel(..., on_result=callback)` (see the
    [Quickstart](../getting-started/quickstart.md)). Your callback gets each
    result in-process, and nothing is written or sent anywhere.

    Sinks are the **advanced** API, for production. Use them when probe output
    has to be delivered **durably** and **out of process**, to a log file that
    a shipper picks up or to an HTTP collector, without slowing down
    generation.

## When to use a sink

A sink receives the output of **async** extraction points: the ones with
`execution_mode: async`, also called "observe mode". In this mode the probe
runs on a background worker, off the generation hot path, so its signals
can't change the response that is being generated. They have to go
somewhere else, and a sink is that somewhere. **Only async bindings are
logged.** An inline extraction point returns its signal synchronously to
whoever called `route()`, and that caller decides what to do with it, so
inline signals and results never reach a sink. See
[Execution modes: inline vs async](../concepts/execution-modes.md) for how
to choose between the two modes.

For each async binding, the router calls two methods on the attached sink:

- `write_signal(request_id, extraction_point_name, signal)` for every
  non-`None` `ProbeSignal` the probe returns from `on_activation`. This runs
  on that binding's worker thread.
- `write_result(request_id, extraction_point_name, result)` once, with the
  final `ProbeResult`, after `on_end`. This runs on the thread that called
  `end_request`, after the binding's queue has been drained.

If a probe's `on_activation` raises, the router logs the exception and
forwards a synthetic signal to the sink: `action="continue"`, with metadata
`{"router_error": true, "error_type": ..., "error": str(exc),
"extraction_point_name": ...}`. A sink is therefore also where probe
failures show up.

Attach a sink with `router.attach_log_sink(sink)`. The equivalent
`undercurrent.sinks.wire_router(router, sink)` does the same thing. The
sink can be attached before or after requests are registered and takes
effect immediately. Pass `None` to detach it.

## `FileLogSink`: NDJSON on local disk

`FileLogSink(path)` appends one JSON object per line
([NDJSON](https://github.com/ndjson/ndjson-spec)) to `path`. Every write
opens the file, appends and closes it again, so there is no handle to manage
and nothing to close. The parent directory must already exist. A lock keeps
lines from different worker threads from interleaving.

The example below sets up a toy async probe and a router, feeds the router
synthetic activations, and then reads the NDJSON back. It doesn't need a
model.

```python
import json
import tempfile
from pathlib import Path

from undercurrent.core import Probe, ProbeAction, ProbeFactory, ProbeResult, ProbeSignal
from undercurrent.router import Router
from undercurrent.sinks import FileLogSink
from undercurrent.spec import ActivationRecord, parse_yaml

SPEC = parse_yaml("""
version: "1"
extraction_points:
  - name: running_mean
    layers: [4]
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: running_mean
    probe_kind: trajectory
    execution_mode: async
    queue_depth: 64
""")


class RunningMeanProbe(Probe):
    """Toy trajectory probe: tracks the mean activation and flags high-scoring tokens."""

    probe_kind = "trajectory"

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def on_start(self, request_ctx):
        self.scores = []
        self.history = []

    def on_activation(self, record):
        score = sum(record.tensor) / len(record.tensor)
        self.scores.append(score)
        if score < self.threshold:
            return None  # nothing worth logging for this token
        signal = ProbeSignal(
            action=ProbeAction.FLAG,
            confidence=score,
            metadata={"token_pos": record.token_pos, "score": score},
        )
        self.history.append(signal)
        return signal

    def on_end(self, request_ctx):
        mean = sum(self.scores) / len(self.scores) if self.scores else None
        return ProbeResult(
            request_id=self.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"mean_score": mean, "flagged_tokens": len(self.history)},
            signal_history=self.history,
            metadata={"threshold": self.threshold},
        )


def fake_activations(request_id, scores):
    """One synthetic ActivationRecord per generated token, standing in for an engine adapter."""
    for pos, score in enumerate(scores):
        yield ActivationRecord(
            request_id=request_id,
            extraction_point_name="running_mean",
            layer=4,
            token_pos=pos,
            tensor_type="residual_stream",
            tensor=[score, score],
            is_generated=True,
        )


workdir = Path(tempfile.mkdtemp())
log_path = workdir / "observations.ndjson"

with Router(probe_registry={"running_mean": ProbeFactory(RunningMeanProbe)}) as router:
    router.attach_log_sink(FileLogSink(log_path))

    with router.request(SPEC, request_id="req-1") as req:
        for record in fake_activations("req-1", [0.1, 0.7, 0.2, 0.9]):
            assert req.route(record) is None  # async: nothing comes back to the caller

records = [json.loads(line) for line in log_path.read_text().splitlines()]
for r in records:
    print(r["kind"], r["request_id"], r["extraction_point_name"], r["payload"].get("metadata"))

assert [r["kind"] for r in records] == ["signal", "signal", "result"]
assert set(records[0]) == {"kind", "request_id", "extraction_point_name", "timestamp", "payload"}
assert records[-1]["payload"]["verdict"] == {"mean_score": 0.475, "flagged_tokens": 2}
```

The code prints:

```console
signal req-1 running_mean {'token_pos': 1, 'score': 0.7}
signal req-1 running_mean {'token_pos': 3, 'score': 0.9}
result req-1 running_mean {'threshold': 0.5}
```

The signals arrive in activation order: each binding has a single worker,
so a probe always sees its activations in order. The result is always the
last record for its binding.

### Record schema

Every built-in sink writes or sends the same record, built by
`undercurrent.sinks.records.build_log_record`:

| Field | Type | Meaning |
| --- | --- | --- |
| `kind` | string | `"signal"` (from `write_signal`) or `"result"` (from `write_result`). |
| `request_id` | string | The request this observation belongs to, as passed to `Router.request` / `register_request` (or chosen by the engine adapter). |
| `extraction_point_name` | string | The `name` of the extraction point in the spec. |
| `timestamp` | number | Unix time in seconds (`time.time()`) when the sink built the record. This is not when the activation happened: see `payload.timestamp` for signals. |
| `payload` | object | The `ProbeSignal` or `ProbeResult`, converted by `undercurrent.sinks.to_jsonable` (see below). |

`payload` for `kind == "signal"` (a `ProbeSignal`):

| Field | Type | Meaning |
| --- | --- | --- |
| `action` | string | `"continue"`, `"flag"` or `"abort"`. Observe-mode signals can't intervene, so `action` is informational here. |
| `metadata` | object | Probe-defined. It can hold anything the probe puts there (see [Redaction and privacy](#redaction-and-privacy)). |
| `confidence` | number or null | Optional probe-defined score. |
| `timestamp` | number | Unix time in seconds when the probe created the signal. |

`payload` for `kind == "result"` (a `ProbeResult`):

| Field | Type | Meaning |
| --- | --- | --- |
| `request_id` | string | Same as the top-level `request_id`. |
| `extraction_point_name` | string | Same as the top-level `extraction_point_name`. |
| `verdict` | any JSON value | The probe's final, probe-defined verdict. |
| `signal_history` | array of signal payloads | Whatever signals the probe chose to keep, in the signal shape above. |
| `metadata` | object | Probe-defined. |

`to_jsonable` makes any payload safe to serialize:

- Dataclasses become objects and enums become their values.
- Dict keys become strings, and tuples and sets become arrays.
- Anything with both `.shape` and `.dtype` (a torch or numpy tensor) is
  **summarized, never dumped**, as
  `{"__summary__": "tensor", "type", "shape", "dtype", "device"?}`.
- `bytes` become `{"__summary__": "bytes", "length": n}`.
- Anything else falls back to `str(value)`.

A probe that stores a raw activation tensor in its metadata therefore logs a
small summary, not megabytes of floats.

## `WebhookLogSink`: HTTP delivery

`WebhookLogSink(url)` POSTs each record as a JSON body
(`Content-Type: application/json`) to `url`.

```py
from undercurrent.sinks import WebhookLogSink

sink = WebhookLogSink(
    "https://example.com/hook",
    max_retries=2,  # 3 attempts in total
    backoff_base=0.1,  # sleeps 0.1 s, then 0.2 s, between attempts
    request_timeout=5.0,  # per-attempt HTTP timeout, in seconds
    dead_letter_path="/var/lib/undercurrent/webhook_dead_letters.ndjson",
    queue_maxsize=1000,  # bounded in-memory backlog
)
router.attach_log_sink(sink)
...
sink.close(timeout=30)  # at shutdown: send what's queued, then stop the worker
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `url` | (required) | Endpoint that receives one `POST` per record. |
| `max_retries` | `2` | Retries after the first attempt, so `max_retries + 1` attempts in total. |
| `backoff_base` | `0.1` | The sleep after failed attempt *n* (counting from 0) is `backoff_base * 2**n` seconds. There's no jitter and no sleep after the last attempt. |
| `request_timeout` | `5.0` | Timeout in seconds for each HTTP attempt. |
| `dead_letter_path` | `None` | NDJSON file that receives records which used up all their attempts. Without one, such a record is logged at `ERROR` level instead. |
| `queue_maxsize` | `1000` | Capacity of the in-memory queue between the router and the sender thread. |
| `post_fn` | `None` | Replaces the HTTP call: `post_fn(url, record)` should raise on failure. Use it for custom auth, a different client, or tests. |
| `redact` | `None` | Extra redaction function, applied **after** the default prompt-text redaction. See [Redaction and privacy](#redaction-and-privacy). |
| `include_prompt_text` | `False` | Set to `True` to turn off the default prompt-text redaction. |

### How delivery works

- **Non-blocking.** `write_signal` and `write_result` only build the record
  and put it on a bounded queue. A single background thread (a daemon thread
  named `undercurrent-sinks-webhook`) does every network call. A slow or
  unreachable endpoint never adds latency to the probe workers or to
  `end_request`.
- **Retries.** Any exception from the POST counts as a failure and is retried
  in the same way: connection errors, timeouts and **any HTTP status of 400
  or above, 4xx included**. A response below 400 counts as delivered.
- **At least once, as long as the process keeps running.** A record is
  retried until it is acknowledged or until it has used up its attempts,
  after which it goes to the dead-letter file. If the endpoint processed a
  request but the response was lost (for example on a timeout), the retry
  delivers the record a second time. Make your collector idempotent: for
  example, de-duplicate on `(request_id, extraction_point_name, kind,
  payload.timestamp)`.
- **Ordering.** A single sender thread sends records in the order they were
  queued. A record that is being retried holds back everything queued after
  it, so retries are strictly head-of-line.
- **Throughput when the endpoint is down.** With the defaults, each record
  takes up to 3 × 5 s of timeouts plus 0.3 s of backoff, about 15 s, before
  it is dead-lettered. The queue fills quickly while the endpoint is down,
  so expect drops (see below).

### Dead letters

A record that fails all `max_retries + 1` attempts is appended to
`dead_letter_path` through an internal `FileLogSink`. Each dead letter is the
same NDJSON record you would have POSTed, **after redaction**, so the
dead-letter file never holds anything the endpoint wouldn't have received.
A `WARNING` is logged each time a record is dead-lettered. The warning names
the URL and the last error, but not the record.

Without a `dead_letter_path`, the record is logged at `ERROR` level on the
`undercurrent.sinks.webhook_sink` logger, with its full (redacted) contents.
For production, set a `dead_letter_path`.

!!! warning "The dead-letter directory must exist and be writable"
    If writing the dead-letter file fails (for example, the directory is
    missing or the disk is full), the record is lost: an `ERROR` naming the
    file and the error is logged, and the sender moves on to the next record.
    Create the directory before you construct the sink, and alert on that
    log message.

### Dropped records and shutdown

The webhook sink loses records in exactly these cases:

| When | What happens | How to see it |
| --- | --- | --- |
| The queue is full when `write_*` is called | The **new** record is discarded and the queued ones are kept. | `sink.dropped_count` goes up. A `WARNING` "queue full ... dropping record" is logged. |
| The `redact` function raises or returns something other than a dict | The record is discarded rather than sent unredacted (fail closed). | `sink.redaction_error_count` goes up. A `WARNING` is logged with the exception type only. |
| The `redact` function returns `None` | The record is discarded on purpose. | Not counted. |
| The process exits before the queue has drained | Records still in the queue are lost. The sender is a daemon thread and doesn't keep the process alive. | Nothing is logged. Call `close()` at shutdown. |
| Writing the dead-letter file fails | The record is discarded; the sender keeps going. | An `ERROR` "could not write to dead-letter file ..." is logged. |
| `close(timeout=...)` times out | `close` returns, and the records that are still queued are lost when the process exits. | `close` gives no indication. Size the timeout to your backlog. |

`close(timeout=None)` queues a stop marker **behind** every record already
queued, then waits up to `timeout` seconds for the sender thread to finish.
Everything queued before `close` is therefore sent or dead-lettered first;
`close` is a flush, not a cancel. Two details:

- Placing the stop marker waits for space in the queue, so a completely full
  queue makes `close` wait until the sender frees a slot. With a `timeout`,
  that wait counts against it, and `close` always returns within `timeout`.
- Don't write to a closed sink. Records written after `close` are queued, but
  nothing will ever send them.

Shut down in this order, so that the router's final results reach the sink
before it stops: `router.shutdown()` (or leave the `with Router(...)` block),
then `sink.close(timeout=...)`.

### Try it locally

This example starts a throwaway HTTP collector on `127.0.0.1` inside the
process. It sends the toy run from above to the collector, then shows a
failing endpoint sending a record to the dead-letter file.

```python
import http.server
import threading

from undercurrent.sinks import WebhookLogSink

received = []


class Collector(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/broken":
            self.send_response(503)  # simulate an outage
        else:
            received.append(json.loads(body))
            self.send_response(204)
        self.end_headers()

    def log_message(self, *args):  # keep the example's output quiet
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
threading.Thread(target=server.serve_forever, daemon=True).start()
base_url = f"http://127.0.0.1:{server.server_port}"

# 1. Healthy endpoint: everything arrives.
sink = WebhookLogSink(f"{base_url}/hook", dead_letter_path=workdir / "dead_letters.ndjson")
with Router(probe_registry={"running_mean": ProbeFactory(RunningMeanProbe)}) as router:
    router.attach_log_sink(sink)
    with router.request(SPEC, request_id="req-2") as req:
        for record in fake_activations("req-2", [0.1, 0.7, 0.2, 0.9]):
            req.route(record)
sink.close(timeout=10)  # flush: returns once the queue is empty

print([r["kind"] for r in received])
assert [r["kind"] for r in received] == ["signal", "signal", "result"]
assert sink.dropped_count == 0

# 2. Failing endpoint: 2 attempts (fast backoff for the demo), then dead-lettered.
broken = WebhookLogSink(
    f"{base_url}/broken",
    max_retries=1,
    backoff_base=0.01,
    dead_letter_path=workdir / "dead_letters.ndjson",
)
broken.write_signal("req-3", "running_mean", ProbeSignal(metadata={"prompt": "my secret", "score": 0.9}))
broken.close(timeout=10)

dead = [json.loads(line) for line in (workdir / "dead_letters.ndjson").read_text().splitlines()]
print(dead[0]["payload"]["metadata"])
assert dead[0]["payload"]["metadata"] == {"prompt": "[REDACTED]", "score": 0.9}

server.shutdown()
```

The code prints the following, plus a `WARNING` on stderr about giving up on
`/broken`:

```console
['signal', 'signal', 'result']
{'prompt': '[REDACTED]', 'score': 0.9}
```

The dead letter is already redacted: the webhook sink redacts the prompt
text before the first attempt, so the plaintext never reaches the
dead-letter file either.

## Redaction and privacy

Observation records hold whatever your probes put in them. The library never
writes prompt or generated text into a record itself, but a probe can do it
with one line. Every probe receives
`RequestContext.prompt_metadata = {"model", "prompt", "prompt_len"}` in
`on_start`, and probes that decode tokens often keep keys like `text` or
`generated_text`. Token ids are text in another encoding. Sending records
to a webhook means **sending data derived from user prompts off the
machine**, to a system with its own retention, access control and
jurisdiction. Decide what leaves on purpose, and see
[Privacy & data handling](../about/privacy.md) for the wider picture.

### What the defaults do

| Sink | Default |
| --- | --- |
| `WebhookLogSink` | **Redacts prompt text.** Every key in `DEFAULT_PROMPT_TEXT_KEYS`, at any depth, has its value replaced by `"[REDACTED]"`. The keys are `prompt`, `prompts`, `prompt_text`, `text`, `input_text`, `output_text`, `generated_text`, `completion`, `response`, `messages`, `input_ids`, `output_ids` and `token_ids`. This applies to dead letters too. Turn it off with `include_prompt_text=True`. When constructed, the sink logs one `INFO` line with the policy in force; the line shows only the scheme and host, never path or query tokens. |
| `FileLogSink` | **No redaction.** Records are written as they are, because the data stays on the local machine. Pass `redact=` if the file is shipped elsewhere, or if the disk isn't trusted with prompt text. |

What the default does **not** cover:

- Keys with other names (`user_id`, `query`, `doc`, and so on).
- Text inside the probe-defined `verdict` when it isn't under one of the
  listed keys.
- The router's synthetic error signal, whose `metadata["error"]` is
  `str(exc)`. If a probe's exception message quotes its input, that input
  goes out. Add `redact_keys("error")` if your probes might do that.
- `request_id` and `extraction_point_name`, which are sent as they are.

### The `redact` hook

Each built-in sink takes `redact=fn`, a function
`(record: dict) -> dict | None` that is applied to the fully serialized
record just before it is written or sent:

- It receives a **deep copy** of the record, so it may modify its argument in
  place.
- If it returns `None`, the record is dropped on purpose.
- If it **raises or returns something other than a dict**, the record is
  dropped (fail closed: it is never sent unredacted), and the sink's
  `redaction_error_count` goes up. The `WARNING` that is logged contains the
  exception type only, never the record's contents.
- `WebhookLogSink` runs it on its sender thread, after the default
  prompt-text redaction. Your function therefore never sees the raw prompt
  text, and a slow function doesn't slow the router down.
- `FileLogSink` runs it inline on the thread that writes the record. Keep it
  cheap.

The helpers in `undercurrent.sinks` cover the common cases:

- `redact_keys(*keys, replacement="[REDACTED]")` replaces the values and keeps
  the keys, so consumers can see that something was removed.
- `drop_keys(*keys)` removes the matching keys entirely.
- `chain(*fns)` applies functions left to right and stops at the first one
  that returns `None`.

Key matching works as follows:

- A plain key (`"prompt"`) matches at any depth.
- A dotted key (`"metadata.raw_scores"`) matches that chain of dict keys
  wherever it appears in the record, for example
  `payload.metadata.raw_scores`.
- Lists are walked through, so `"signal_history.metadata.prompt"` matches
  inside every element of a result's history.
- Matching is exact and case-sensitive: `prompt` does not match
  `prompt_len`.

### Recommended production config: allowlist what leaves

A denylist only catches the key names you thought of. For data that leaves
the machine, it's safer to **allowlist** the metadata fields your collector
actually needs and drop everything else. The library doesn't ship an
allowlist helper, but you can write one as a plain `redact` function:

```python
from undercurrent.sinks import DEFAULT_PROMPT_TEXT_KEYS, chain, drop_keys, redact_keys


def allow_metadata_keys(*allowed):
    """Keep only the listed keys in every `metadata` object (the signal's, the result's,
    and each signal in `signal_history`), and keep only scalar verdicts."""
    allowed = set(allowed)

    def _redact(record):
        payload = record["payload"]
        for item in [payload, *payload.get("signal_history", [])]:
            if isinstance(item.get("metadata"), dict):
                item["metadata"] = {k: v for k, v in item["metadata"].items() if k in allowed}
        if isinstance(payload.get("verdict"), (dict, list, str)):
            payload["verdict"] = "[REDACTED]"
        return record

    return _redact


production_redact = chain(
    redact_keys(*DEFAULT_PROMPT_TEXT_KEYS),  # belt and braces: also covers keys inside a verdict
    drop_keys("error"),  # exception messages can quote user input
    allow_metadata_keys("score", "token_pos", "threshold", "router_error", "error_type"),
)

# Check it on a record that a careless probe has filled with prompt text:
check_path = workdir / "redacted.ndjson"
check_sink = FileLogSink(check_path, redact=production_redact)
check_sink.write_signal(
    "req-4",
    "running_mean",
    ProbeSignal(metadata={"score": 0.9, "prompt": "my secret", "user_email": "a@example.com"}),
)
check_sink.write_result(
    "req-4",
    "running_mean",
    ProbeResult("req-4", "running_mean", verdict={"summary": "my secret"}, metadata={"threshold": 0.5}),
)

signal_rec, result_rec = [json.loads(line) for line in check_path.read_text().splitlines()]
print(signal_rec["payload"]["metadata"], result_rec["payload"]["verdict"])
assert signal_rec["payload"]["metadata"] == {"score": 0.9}
assert "my secret" not in check_path.read_text()
assert check_sink.redaction_error_count == 0
```

The code prints:

```console
{'score': 0.9} [REDACTED]
```

The `verdict` handling is deliberately blunt: adjust it to the verdict
shapes your probes actually produce. In production, plug the same function
into the webhook. Keep the default prompt-text redaction on (don't pass
`include_prompt_text=True`); your function then runs after it:

```py
sink = WebhookLogSink(
    "https://example.com/hook",
    dead_letter_path="/var/lib/undercurrent/webhook_dead_letters.ndjson",
    redact=chain(
        drop_keys("error"),
        allow_metadata_keys("score", "token_pos", "threshold", "router_error", "error_type"),
    ),
)
```

Other things to consider:

- If `request_id` is linkable to a user in your system, replace it with a
  keyed hash inside your `redact` function.
- Use HTTPS. Put credentials in a header through `post_fn`, not in the URL.
  The sink never logs URL paths or queries at construction, but the warning
  logged when a record is dead-lettered includes the full URL.
- Treat the dead-letter file and NDJSON logs as data with the same
  sensitivity as whatever you allowed through.

## Writing your own sink

To write your own sink, subclass `LogSink` and implement `write_signal` and
`write_result`. Any object with those two methods works with
`attach_log_sink`, but subclassing gives you the `redact=` support. The
router relies on this contract:

- **Thread safety.** `write_signal` is called concurrently from every async
  binding's worker thread. `write_result` is called from whichever thread
  calls `end_request`. Protect shared state with a lock.
- **Don't block.** `write_signal` runs on the binding's worker thread, so
  time spent in it is time that binding isn't draining its queue. A slow
  sink grows the queue, and once the queue is full the overflow policy
  starts dropping activations (see
  [Async execution & backpressure](../production/async-execution.md)).
  `write_result` blocks `end_request`. A fast local append is fine. Hand
  network I/O to a background thread with a bounded queue, the way
  `WebhookLogSink` does.
- **Don't raise.** The router catches and logs any exception from a sink, so
  a broken sink can't crash generation. The record is lost, though, and you
  get a stack trace for every record. Handle your own errors.
- **Serialize immediately.** The `signal` and `result` objects belong to the
  probe. Convert them with `to_jsonable` at call time, and don't keep
  references or mutate them.

This sink prints compact JSON to stdout and supports `redact=`:

```python
import sys
import time

from undercurrent.sinks import LogSink, to_jsonable


class StdoutSink(LogSink):
    def __init__(self, *, redact=None):
        super().__init__(redact=redact)
        self._lock = threading.Lock()

    def write_signal(self, request_id, extraction_point_name, signal):
        self._emit("signal", request_id, extraction_point_name, signal)

    def write_result(self, request_id, extraction_point_name, result):
        self._emit("result", request_id, extraction_point_name, result)

    def _emit(self, kind, request_id, extraction_point_name, payload):
        try:
            record = {
                "kind": kind,
                "request_id": request_id,
                "extraction_point_name": extraction_point_name,
                "timestamp": time.time(),
                "payload": to_jsonable(payload),
            }
            record = self._apply_redaction(record)  # None means "drop it"
            if record is None:
                return
            line = json.dumps(record, default=str)
            with self._lock:
                sys.stdout.write(line + "\n")
        except Exception:  # never raise into the router
            pass


with Router(probe_registry={"running_mean": ProbeFactory(RunningMeanProbe)}) as router:
    router.attach_log_sink(StdoutSink(redact=drop_keys("signal_history")))
    with router.request(SPEC, request_id="req-5") as req:
        for record in fake_activations("req-5", [0.9]):
            req.route(record)
```

To decouple delivery completely, push records onto an in-process queue and
let your own consumer (for example a Kafka producer or a batching uploader)
drain it. Use `put_nowait`, and count drops instead of blocking:

```python
import queue


class QueueSink(LogSink):
    def __init__(self, maxsize=10_000):
        super().__init__()
        self.queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._lock = threading.Lock()

    def write_signal(self, request_id, extraction_point_name, signal):
        self._put(("signal", request_id, extraction_point_name, to_jsonable(signal)))

    def write_result(self, request_id, extraction_point_name, result):
        self._put(("result", request_id, extraction_point_name, to_jsonable(result)))

    def _put(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.dropped += 1


qsink = QueueSink()
with Router(probe_registry={"running_mean": ProbeFactory(RunningMeanProbe)}) as router:
    router.attach_log_sink(qsink)
    with router.request(SPEC, request_id="req-6") as req:
        for record in fake_activations("req-6", [0.1, 0.8]):
            req.route(record)

print([item[0] for item in qsink.queue.queue])
assert [item[0] for item in qsink.queue.queue] == ["signal", "result"]
```

## Operational tips

**File rotation.** `FileLogSink` doesn't rotate or cap its file. Because it
reopens the path on every write, an external rotator works without
`copytruncate`: once the file has been renamed, the next write creates a
fresh file at the original path. A `logrotate` stanza:

```text
/var/log/undercurrent/observations.ndjson {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
}
```

Records written between the rename and the next write go to the new file.
No lines are lost or split. To rotate by size, or inside the process, write
a custom sink (see above) or ship the file with a log forwarder that handles
rotation (Vector, Fluent Bit, the OpenTelemetry Collector's filelog
receiver).

**Monitor the webhook sink.** Its health isn't part of the router's
metrics, so watch these yourself:

- The **dead-letter file**: its size or line count. Any growth means the
  endpoint rejected records or was unreachable. Alert on growth, not on the
  absolute size. Once the endpoint is healthy, you can replay dead letters
  by POSTing each line again. They're already redacted, so a replay never
  sends more than the original attempt would have.
- `sink.dropped_count`: records lost because the queue was full. A non-zero
  rate means the endpoint can't keep up, or the sender thread has stopped
  (see the dead-letter warning above).
- `sink.redaction_error_count`: records dropped by a failing `redact`
  function, which is almost always a bug in that function.
- `WARNING`/`ERROR` log lines from the `undercurrent.sinks` loggers.

The two counters are plain properties. Export them as gauges from a
periodic task.

**Router metrics that relate to sinks.** The router's own metrics (see
[Metrics](../production/metrics.md)) don't measure sinks directly, but a
slow sink shows up in them:

- `write_signal` runs on the binding's worker thread after `on_activation`
  returns. Time spent in a slow sink is therefore **not** included in the
  activation latency (`record_activation`, `avg_activation_latency_seconds`),
  but it does keep the worker from draining its queue.
- The symptom is rising **queue depth** (`record_queue_depth`) and then
  **drops** (`record_drop` / `drop_count`) while activation latency looks
  normal. If you see that pattern, check your sink first.
  `WebhookLogSink` and `FileLogSink` return quickly, so a custom sink is the
  usual cause.

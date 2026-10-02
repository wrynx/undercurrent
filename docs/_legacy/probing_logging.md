# probing_logging

Out-of-band **observation logging sink** for the activation-probing
platform. For an extraction point running in *observe mode*
(`execution_mode=async`, as opposed to an inline probe that can intervene
mid-generation), a probe's signals and final verdict shouldn't block the
response -- they're logged here instead.

Built on [`undercurrent.core`](../../src/undercurrent/core/) (`ProbeResult`, `ProbeSignal`)
and [`undercurrent.router`](../../src/undercurrent/router/) (`Router.attach_log_sink`) --
types are imported directly from those packages.

## Install

```bash
pip install -e .   # at the repo root: the undercurrent project (spec, core, router, sinks)
```

Requires Python 3.9+. No third-party runtime dependencies -- `WebhookLogSink`
uses `urllib` from the standard library.

## Usage

```python
from undercurrent.sinks import FileLogSink, WebhookLogSink
from undercurrent.router import Router

router = Router(probe_registry={...})

# Append NDJSON locally:
router.attach_log_sink(FileLogSink("observations.ndjson"))

# Or POST to a webhook, with local dead-lettering if delivery ultimately fails:
sink = WebhookLogSink(
    "https://example.com/hook",
    dead_letter_path="webhook_dead_letters.ndjson",
)
router.attach_log_sink(sink)
# equivalently: undercurrent.sinks.wire_router(router, sink)

router.register_request(request_id, extraction_points, request_ctx)
for record in activation_stream:
    router.route(record)   # async trajectory bindings now also log every signal
results = router.end_request(request_id)   # ...and the final ProbeResult
```

`router.attach_log_sink` only ever forwards from **async** bindings (inline
extraction points already return their signal synchronously to `route()`'s
caller). Every non-`None` `ProbeSignal` an async trajectory probe's
`on_activation` returns is forwarded to `sink.write_signal(...)`, and its
final `ProbeResult` (from `on_end`) is forwarded to `sink.write_result(...)`
once `end_request` finalizes it -- both from the async binding's own worker
thread, never from inside `route()`'s caller, so a slow or failing sink
never adds latency to dispatch (see the router's legacy guide in `docs/_legacy/` for the full
non-blocking contract this package relies on).

## `LogSink` interface

```python
class LogSink(ABC):
    def write_signal(self, request_id: str, extraction_point_name: str, signal: ProbeSignal) -> None: ...
    def write_result(self, request_id: str, extraction_point_name: str, result: ProbeResult) -> None: ...
```

`Router.attach_log_sink` doesn't actually require a `LogSink` subclass --
it only needs something that duck-types these two methods
(`undercurrent.router.SupportsLogSink`) -- but every sink in this package
implements the ABC.

## Concrete sinks

### `FileLogSink`

Appends one newline-delimited JSON (NDJSON) record per signal/result to a
local file:

```python
sink = FileLogSink("observations.ndjson")
```

Each line: `{"kind": "signal"|"result", "request_id", "extraction_point_name",
"timestamp", "payload"}`. `payload` is the `ProbeSignal`/`ProbeResult`,
recursively converted to plain JSON values via
`undercurrent.sinks.serialize.to_jsonable` -- dataclasses/enums are unwrapped
structurally, and anything tensor-like (duck-typed via `.shape`/`.dtype`,
so this works for `torch.Tensor` or `numpy.ndarray` without depending on
either) is **summarized** (`{"__summary__": "tensor", "shape": ..., "dtype":
...}`), never dumped as raw values. A shared lock serializes writes so
concurrent callers (different async bindings, each on their own thread)
never interleave a line.

### `WebhookLogSink`

POSTs the same record as JSON to a configured HTTP endpoint:

```python
sink = WebhookLogSink(
    "https://example.com/hook",
    max_retries=2,           # default: 3 attempts total
    backoff_base=0.1,        # seconds; backoff is backoff_base * 2**attempt
    dead_letter_path="webhook_dead_letters.ndjson",  # NDJSON via FileLogSink; omit to log locally instead
)
```

`write_signal`/`write_result` only enqueue onto an internal bounded queue
and return immediately -- the actual HTTP POST happens on a single
dedicated background worker thread, so a slow or unreachable endpoint never
blocks the caller. A failed POST is retried `max_retries` times with
exponential backoff; once retries are exhausted the record is written to
`dead_letter_path` (or logged via the standard `logging` module if none was
given) rather than lost silently or raised anywhere. If the background
queue itself is ever full (the worker has fallen far behind), new records
are dropped and logged rather than blocking the caller or growing without
bound (`sink.dropped_count` tracks how many).

For tests, `post_fn` can be overridden to avoid real network calls;
`close(timeout=...)` stops the background worker after everything already
queued has been sent/dead-lettered (useful to await completion
deterministically -- the worker thread is otherwise a daemon and doesn't
need explicit shutdown).

## Redaction

Every built-in sink takes `redact=fn`, a `(record: dict) -> dict | None`
applied to the serialised record just before it is written, sent or
dead-lettered (`None` drops it; an exception drops it too and bumps
`sink.redaction_error_count`). Helpers: `redact_keys(...)`, `drop_keys(...)`,
`chain(...)`. `WebhookLogSink` redacts `DEFAULT_PROMPT_TEXT_KEYS` (`prompt`,
`text`, `generated_text`, ...) by default; `include_prompt_text=True` opts
out. `FileLogSink` does not redact by default. The field audit (which record
keys can carry user text) is in the module docstring of
`src/undercurrent/sinks/redaction.py`.

## Serialization (`to_jsonable`)

`undercurrent.sinks.serialize.to_jsonable` is the shared conversion both sinks
use to turn a `ProbeSignal`/`ProbeResult` (or anything nested inside their
`metadata`/`verdict` fields) into something `json.dumps` can always handle:
dataclasses and enums are unwrapped, dicts/lists/tuples/sets recurse,
tensor-like objects are summarized (shape + dtype, not raw data), and
anything else falls back to `str(value)` rather than raising.

## Dependency shim

The stub-fallback dependency shim was removed in the single-package consolidation; shared types are imported directly.

## `undercurrent.router` integration point

`Router.attach_log_sink(sink)` is implemented in `undercurrent.router` itself
(`undercurrent/router/binding.py`'s `LogSinkHolder`/`SupportsLogSink`,
`undercurrent/router/router.py`'s `attach_log_sink`/`end_request`) -- see that
package's README for the mechanism. This package takes a dependency on
`undercurrent.router` for `Router` (used only by the `wire_router` convenience
function and for typing), but `undercurrent.router` takes **no** dependency on
`undercurrent.sinks`: the hook is a structural (duck-typed) interface, not an
import.

## Package layout

```
src/undercurrent/sinks/
  serialize.py    to_jsonable -- recursive JSON-safe conversion, incl. tensor summarization
  records.py      build_log_record -- the shared {kind, request_id, ..., payload} shape
  sink.py         LogSink (abstract interface)
  file_sink.py    FileLogSink
  webhook_sink.py WebhookLogSink
  redaction.py    redact_keys / drop_keys / chain, DEFAULT_PROMPT_TEXT_KEYS, field audit
  wiring.py       wire_router convenience function
tests/sinks/      unit tests (pytest)
```

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest tests/sinks
```

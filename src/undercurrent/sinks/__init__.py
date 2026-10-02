"""undercurrent.sinks: out-of-band observation logging for the
activation-probing platform.

For an extraction point running in *observe mode* (`execution_mode=async`,
as opposed to an inline probe that can intervene mid-generation), a probe's
signals and final verdict don't block the response -- they're logged here
instead. Built on `undercurrent.core` (`ProbeResult`, `ProbeSignal`) and
`undercurrent.router` (`Router.attach_log_sink`).

Typical usage::

    from undercurrent.sinks import FileLogSink, WebhookLogSink

    sink = FileLogSink("observations.ndjson")
    router.attach_log_sink(sink)   # undercurrent.router.Router

    # or, layer a webhook with a local dead-letter fallback:
    sink = WebhookLogSink("https://example.com/hook", dead_letter_path="webhook_dead_letters.ndjson")
    router.attach_log_sink(sink)

    # control what leaves the process (WebhookLogSink redacts prompt text by default):
    sink = WebhookLogSink(url, redact=chain(drop_keys("metadata.raw_scores"), redact_keys("user_id")))

Public surface:
    - Interface: `LogSink`
    - Concrete sinks: `FileLogSink`, `WebhookLogSink`
    - Wiring: `wire_router()` (equivalent to `router.attach_log_sink(sink)`)
    - Redaction: `redact_keys()`, `drop_keys()`, `chain()`,
      `DEFAULT_PROMPT_TEXT_KEYS`, `RedactFn`
    - For custom sinks: `to_jsonable()` (converts probe payloads --
      dataclasses, enums, tensors -- into JSON-safe values)

The modules `records`, `serialize` (apart from `to_jsonable`) and `wiring`
(apart from `wire_router`) are internal.
"""

from .file_sink import FileLogSink
from .redaction import DEFAULT_PROMPT_TEXT_KEYS, RedactFn, chain, drop_keys, redact_keys
from .serialize import to_jsonable
from .sink import LogSink
from .webhook_sink import WebhookLogSink
from .wiring import wire_router

__all__ = [
    "DEFAULT_PROMPT_TEXT_KEYS",
    "FileLogSink",
    "LogSink",
    "RedactFn",
    "WebhookLogSink",
    "chain",
    "drop_keys",
    "redact_keys",
    "to_jsonable",
    "wire_router",
]

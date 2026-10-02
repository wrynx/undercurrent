"""WebhookLogSink: POSTs signals/results to an HTTP endpoint, off the caller's thread."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core import ProbeResult, ProbeSignal
from .file_sink import FileLogSink
from .records import RESULT, SIGNAL, build_log_record
from .redaction import DEFAULT_PROMPT_TEXT_KEYS, RedactFn, chain, redact_keys
from .sink import LogSink

logger = logging.getLogger(__name__)

PostFn = Callable[[str, dict[str, Any]], None]

_STOP = object()


def _default_post(url: str, record: dict[str, Any], timeout: float) -> None:
    data = json.dumps(record, default=str).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status >= 400:
            raise urllib.error.HTTPError(
                url, response.status, "webhook returned an error status", response.headers, None
            )


def _describe_endpoint(url: str) -> str:
    """Scheme + host only, so a token in the URL's path/query/userinfo never reaches the log."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname or ""
        return f"{parts.scheme}://{host}" if parts.scheme else host or "<url>"
    except ValueError:
        return "<url>"


class WebhookLogSink(LogSink):
    """POSTs each signal/result as JSON to ``url`` from a background thread.

    ``write_signal`` / ``write_result`` only enqueue and return, so a slow or
    unreachable endpoint never adds latency to the caller.

    A failed POST is retried ``max_retries`` times (3 attempts in total by
    default) with exponential backoff (``backoff_base * 2**attempt``
    seconds). A record that exhausts its retries is written to
    ``dead_letter_path`` (NDJSON) if one was given, and otherwise logged with
    the standard ``logging`` module. Errors are never raised to the caller.
    If the bounded queue fills up (the endpoint is down and records are
    waiting on retries), new records are dropped and counted in
    [`dropped_count`][undercurrent.sinks.WebhookLogSink.dropped_count].

    Because records leave the machine, every key in
    [`DEFAULT_PROMPT_TEXT_KEYS`][undercurrent.sinks.DEFAULT_PROMPT_TEXT_KEYS]
    (``prompt``, ``text``, ``generated_text``, ...) is replaced with
    ``"[REDACTED]"`` by default, wherever it appears. Redaction runs before
    the first send attempt, so the dead-letter file and the local error log
    only ever see the redacted record.

    Args:
        url: the endpoint to POST to.
        max_retries: retries after the first failed attempt.
        backoff_base: base of the exponential backoff, in seconds.
        request_timeout: per-request timeout, in seconds.
        dead_letter_path: NDJSON file for records that exhaust their retries.
        queue_maxsize: capacity of the background queue.
        post_fn: replaces the HTTP POST: called as ``post_fn(url, record)``
            and expected to raise on failure. Useful for tests or a custom
            HTTP client.
        redact: a custom [`RedactFn`][undercurrent.sinks.RedactFn], run after
            the default prompt-text redaction.
        include_prompt_text: send the ``DEFAULT_PROMPT_TEXT_KEYS`` as-is (a
            custom ``redact`` then runs alone).
    """

    def __init__(
        self,
        url: str,
        *,
        max_retries: int = 2,
        backoff_base: float = 0.1,
        request_timeout: float = 5.0,
        dead_letter_path: str | Path | None = None,
        queue_maxsize: int = 1000,
        post_fn: PostFn | None = None,
        redact: RedactFn | None = None,
        include_prompt_text: bool = False,
    ) -> None:
        if include_prompt_text:
            effective_redact = redact
            policy = "prompt text is INCLUDED in sent records (include_prompt_text=True)"
        else:
            default_redact = redact_keys(*DEFAULT_PROMPT_TEXT_KEYS)
            effective_redact = default_redact if redact is None else chain(default_redact, redact)
            policy = (
                f"prompt-text keys ({', '.join(DEFAULT_PROMPT_TEXT_KEYS)}) are redacted before sending; "
                "pass include_prompt_text=True to send them"
            )
        super().__init__(redact=effective_redact)
        logger.info(
            "WebhookLogSink(%s): %s; custom redact function: %s",
            _describe_endpoint(url),
            policy,
            "yes" if redact is not None else "none",
        )
        self._url = url
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._post_fn: PostFn = post_fn or (
            lambda target_url, record: _default_post(target_url, record, request_timeout)
        )
        self._dead_letter = FileLogSink(dead_letter_path) if dead_letter_path is not None else None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_maxsize)
        self._dropped_count = 0
        self._dropped_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="undercurrent-sinks-webhook", daemon=True)
        self._thread.start()

    @property
    def dropped_count(self) -> int:
        """Records discarded because the background queue was full. For tests/introspection."""
        with self._dropped_lock:
            return self._dropped_count

    def write_signal(self, request_id: str, extraction_point_name: str, signal: ProbeSignal) -> None:
        self._enqueue(build_log_record(SIGNAL, request_id, extraction_point_name, signal))

    def write_result(self, request_id: str, extraction_point_name: str, result: ProbeResult) -> None:
        self._enqueue(build_log_record(RESULT, request_id, extraction_point_name, result))

    def _enqueue(self, record: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._dropped_lock:
                self._dropped_count += 1
            logger.warning("WebhookLogSink queue full (maxsize=%d) -- dropping record", self._queue.maxsize)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                self._send_with_retry(item)
            except Exception:  # noqa: BLE001 -- the sender thread must outlive any one record
                # E.g. the dead-letter file is unwritable. Without this, the
                # thread would die, later records would pile up unsent, and
                # close() would wait on a thread that's gone.
                logger.exception("WebhookLogSink: failed to deliver or dead-letter a record; dropping it")

    def _send_with_retry(self, record: dict[str, Any]) -> None:
        redacted = self._apply_redaction(record)
        if redacted is None:
            return
        record = redacted
        for attempt in range(self._max_retries + 1):
            try:
                self._post_fn(self._url, record)
                return
            except Exception as exc:  # noqa: BLE001 -- any failure (network, HTTP status, ...) is retryable the same way
                is_last_attempt = attempt == self._max_retries
                if is_last_attempt:
                    logger.warning(
                        "WebhookLogSink: giving up on %s after %d attempt(s) (%s); writing to dead letter",
                        self._url,
                        attempt + 1,
                        exc,
                    )
                    self._dead_letter_write(record)
                    return
                time.sleep(self._backoff_base * (2**attempt))

    def _dead_letter_write(self, record: dict[str, Any]) -> None:
        if self._dead_letter is not None:
            try:
                self._dead_letter.write_raw(record)
            except OSError as exc:
                logger.error(
                    "WebhookLogSink: could not write to dead-letter file %s (%s); dropping record",
                    self._dead_letter.path,
                    exc,
                )
        else:
            logger.error("WebhookLogSink: permanent delivery failure, dropping record: %r", record)

    def close(self, timeout: float | None = None) -> None:
        """Stop the background worker after everything already queued has
        been sent (or dead-lettered). Mainly for tests/clean shutdown --
        the worker thread is a daemon, so it won't keep the process alive
        on its own.

        With a ``timeout`` (seconds), returns within it even if records are
        still being sent; without one, waits for the queue to drain."""
        if not self._thread.is_alive():
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            # A full queue would otherwise block this put past the timeout.
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            logger.warning("WebhookLogSink.close(): queue still full after %.1fs; not waiting for it", timeout)
            return
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        self._thread.join(timeout=remaining)

"""LogSink: the interface every concrete observation-logging backend implements."""

from __future__ import annotations

import copy
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

from ..core import ProbeResult, ProbeSignal
from .redaction import RedactFn

logger = logging.getLogger(__name__)


class LogSink(ABC):
    """Out-of-band destination for a probe's intermediate signals and final verdict.

    A router calls a sink for extraction points in observe mode
    (``execution_mode=async``), whose output shouldn't block the response.
    Attach one with
    [`Router.attach_log_sink`][undercurrent.router.Router.attach_log_sink]
    (or ``ProbedModel(log_sink=...)``). The router calls ``write_signal`` /
    ``write_result`` from an async binding's own worker thread, so neither
    should block for long: a fast local append is fine, but a network call
    should hand off to a background thread, as
    [`WebhookLogSink`][undercurrent.sinks.WebhookLogSink] does.

    To write a custom sink, subclass ``LogSink``, implement the two abstract
    methods, and convert payloads with
    [`to_jsonable`][undercurrent.sinks.to_jsonable]. To support a ``redact=``
    argument, call ``super().__init__(redact=redact)`` and pass every record
    through ``self._apply_redaction(record)`` just before writing it (it
    returns ``None`` when the record must be dropped).

    Args:
        redact: an optional [`RedactFn`][undercurrent.sinks.RedactFn] applied
            to each record before it is written.
    """

    _redaction_counter_lock = threading.Lock()

    def __init__(self, *, redact: RedactFn | None = None) -> None:
        self._redact = redact
        self._redaction_error_count = 0

    @property
    def redaction_error_count(self) -> int:
        """Records dropped because the ``redact`` function raised (or returned a non-dict)."""
        return getattr(self, "_redaction_error_count", 0)

    def _apply_redaction(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Run this sink's ``redact`` function on a deep copy of `record`.

        Returns the record to write, or ``None`` to drop it. Fails closed: if
        ``redact`` raises or returns something other than a dict or ``None``,
        the error is logged (without the record's contents), counted in
        `redaction_error_count`, and the record is dropped.
        """
        redact = getattr(self, "_redact", None)
        if redact is None:
            return record
        try:
            redacted = redact(copy.deepcopy(record))
        except Exception as exc:  # noqa: BLE001 -- any failure must drop the record, never leak it
            self._count_redaction_error()
            logger.warning(
                "%s: redact function raised %s; dropping record (extraction_point_name=%r)",
                type(self).__name__,
                type(exc).__name__,
                record.get("extraction_point_name") if isinstance(record, dict) else None,
            )
            logger.debug("redact function traceback", exc_info=True)
            return None
        if redacted is not None and not isinstance(redacted, dict):
            self._count_redaction_error()
            logger.warning(
                "%s: redact function returned %s (expected dict or None); dropping record",
                type(self).__name__,
                type(redacted).__name__,
            )
            return None
        return redacted

    def _count_redaction_error(self) -> None:
        with LogSink._redaction_counter_lock:
            self._redaction_error_count = getattr(self, "_redaction_error_count", 0) + 1

    @abstractmethod
    def write_signal(self, request_id: str, extraction_point_name: str, signal: ProbeSignal) -> None:
        """Called whenever a probe emits an intermediate `ProbeSignal`,
        e.g. a trajectory probe's running-score update from `on_activation`."""

    @abstractmethod
    def write_result(self, request_id: str, extraction_point_name: str, result: ProbeResult) -> None:
        """Called once, after a probe's `on_end` finalizes its `ProbeResult`."""

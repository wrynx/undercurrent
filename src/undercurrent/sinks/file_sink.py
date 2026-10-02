"""FileLogSink: appends newline-delimited JSON (NDJSON) to a log file."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..core import ProbeResult, ProbeSignal
from .records import RESULT, SIGNAL, build_log_record
from .redaction import RedactFn
from .sink import LogSink


class FileLogSink(LogSink):
    """Appends one NDJSON line per signal/result to ``path``.

    Each line is a JSON object with ``kind`` (``"signal"``/``"result"``),
    ``request_id``, ``extraction_point_name``, ``timestamp`` and ``payload``
    (the signal or result, converted with
    [`to_jsonable`][undercurrent.sinks.to_jsonable], so tensors are
    summarized rather than dumped).

    Writes are serialized, so lines from different async bindings never
    interleave. Each write opens, appends to and closes the file, so there
    is nothing to close; ``path``'s parent directory must exist.

    Unlike [`WebhookLogSink`][undercurrent.sinks.WebhookLogSink], a
    ``FileLogSink`` keeps data on the local machine and doesn't redact prompt
    text by default.

    Args:
        path: the NDJSON file to append to.
        redact: an optional [`RedactFn`][undercurrent.sinks.RedactFn] applied
            to each record just before it is written.
    """

    def __init__(self, path: str | Path, *, redact: RedactFn | None = None) -> None:
        super().__init__(redact=redact)
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """The file records are appended to."""
        return self._path

    def write_signal(self, request_id: str, extraction_point_name: str, signal: ProbeSignal) -> None:
        self._write(build_log_record(SIGNAL, request_id, extraction_point_name, signal))

    def write_result(self, request_id: str, extraction_point_name: str, result: ProbeResult) -> None:
        self._write(build_log_record(RESULT, request_id, extraction_point_name, result))

    def write_raw(self, record: dict[str, Any]) -> None:
        """Append an already-built record (still passed through ``redact``, if set).

        ``WebhookLogSink`` uses this for its dead-letter file.
        """
        self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        redacted = self._apply_redaction(record)
        if redacted is None:
            return
        record = redacted
        line = json.dumps(record, default=str)
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

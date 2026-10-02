"""Convenience wiring between an `undercurrent.router.Router` and a `LogSink`."""

from __future__ import annotations

from ..router import Router
from .sink import LogSink


def wire_router(router: Router, sink: LogSink) -> None:
    """Attach ``sink`` to ``router``; the same as ``router.attach_log_sink(sink)``.

    Every async ("observe mode") binding's signals and final results are then
    forwarded to ``sink``. See
    [`Router.attach_log_sink`][undercurrent.router.Router.attach_log_sink].
    """
    router.attach_log_sink(sink)

"""Observability for async bindings: per-(request_id, extraction_point_name)
queue depth, drop count, on_activation latency, and probe-error count.

Modeled after `undercurrent.sinks.LogSink`, one level down the dependency
stack: `MetricsSink` is a small push-style ABC any backend (Prometheus,
StatsD, ...) could implement and plug into `Router.attach_metrics_sink`,
mirroring `Router.attach_log_sink`'s contract exactly (forwarding happens
from the async binding's own worker thread, never blocks or affects
dispatch, a raising sink is caught and logged). Unlike `LogSink`, no
dependency-direction concern requires this to live in a separate package or
be duck-typed structurally -- there is nothing upstream of `undercurrent.router`
that would need to depend on it -- so it's defined here directly as a
concrete ABC.

`InMemoryMetricsRegistry` is the built-in, always-on implementation:
`Router` constructs one at startup and answers `Router.get_metrics(...)`
from it regardless of whether an external sink is also attached, so
querying metrics works out of the box with no wiring required. It is not
swappable (it isn't part of the pluggable-backend story); it exists purely
so this package satisfies "queryable metrics" without requiring a caller to
provide a backend.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MetricsSink(ABC):
    """Pluggable destination for async-binding metrics events.

    Subclass it to export metrics (Prometheus, StatsD, ...) and attach it with
    ``Router(metrics_sink=...)`` or ``Router.attach_metrics_sink``. Every
    method is called from the async binding's own worker thread, never from
    ``Router.route()``'s caller, so a slow or raising sink only delays that
    binding's processing, never dispatch.
    """

    @abstractmethod
    def record_queue_depth(self, request_id: str, extraction_point_name: str, depth: int) -> None:
        """Current number of items sitting in the binding's queue, sampled
        immediately after it changes (a `route()` enqueue or a worker
        dequeue)."""

    @abstractmethod
    def record_drop(self, request_id: str, extraction_point_name: str) -> None:
        """Called once per item the queue's overflow policy discarded --
        either an eviction (drop_oldest) or a rejection (drop_newest).
        Never called for a `put()` rejected merely because the queue was
        already closed; that's shutdown behavior, not overflow."""

    @abstractmethod
    def record_activation(self, request_id: str, extraction_point_name: str, latency_seconds: float) -> None:
        """Called once per `on_activation` call the worker made (whether it
        returned normally or raised), with the wall-clock time it took."""

    @abstractmethod
    def record_probe_error(self, request_id: str, extraction_point_name: str) -> None:
        """Called once per `on_activation` call that raised."""


@dataclass(frozen=True)
class MetricsSnapshot:
    """Point-in-time read of one async binding's metrics.

    Attributes:
        queue_depth: activations waiting in the binding's queue.
        drop_count: activations the overflow policy discarded.
        activation_count: ``on_activation`` calls made (including ones that raised).
        error_count: ``on_activation`` calls that raised.
        avg_activation_latency_seconds: mean ``on_activation`` wall-clock
            time, or None before the first call.
    """

    queue_depth: int
    drop_count: int
    activation_count: int
    error_count: int
    avg_activation_latency_seconds: float | None


@dataclass
class _MutableStats:
    queue_depth: int = 0
    drop_count: int = 0
    activation_count: int = 0
    activation_latency_total: float = 0.0
    error_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class InMemoryMetricsRegistry(MetricsSink):
    """Thread-safe in-memory counters and gauges, keyed by (request_id, extraction_point_name).

    Every ``Router`` keeps one; ``Router.get_metrics`` reads it. Updates to
    different bindings never contend with each other.
    """

    # Each key gets its own _MutableStats and lock, created lazily under a
    # short-held dict lock, so only concurrent updates to the same binding contend
    # (at most one worker plus occasional route()-side queue-depth updates).

    def __init__(self) -> None:
        self._stats: dict[tuple[str, str], _MutableStats] = {}
        self._dict_lock = threading.Lock()

    def register(self, request_id: str, extraction_point_name: str) -> None:
        """Seed a zeroed entry, so a query before the first activation returns zeros."""
        self._get_or_create(request_id, extraction_point_name)

    def forget(self, request_id: str, extraction_point_name: str) -> None:
        """Drop the entry for one binding (``end_request`` does this, so metrics don't outlive the request)."""
        with self._dict_lock:
            self._stats.pop((request_id, extraction_point_name), None)

    def _get_or_create(self, request_id: str, extraction_point_name: str) -> _MutableStats:
        key = (request_id, extraction_point_name)
        with self._dict_lock:
            stats = self._stats.get(key)
            if stats is None:
                stats = _MutableStats()
                self._stats[key] = stats
            return stats

    def record_queue_depth(self, request_id: str, extraction_point_name: str, depth: int) -> None:
        stats = self._get_or_create(request_id, extraction_point_name)
        with stats.lock:
            stats.queue_depth = depth

    def record_drop(self, request_id: str, extraction_point_name: str) -> None:
        stats = self._get_or_create(request_id, extraction_point_name)
        with stats.lock:
            stats.drop_count += 1

    def record_activation(self, request_id: str, extraction_point_name: str, latency_seconds: float) -> None:
        stats = self._get_or_create(request_id, extraction_point_name)
        with stats.lock:
            stats.activation_count += 1
            stats.activation_latency_total += latency_seconds

    def record_probe_error(self, request_id: str, extraction_point_name: str) -> None:
        stats = self._get_or_create(request_id, extraction_point_name)
        with stats.lock:
            stats.error_count += 1

    def snapshot(self, request_id: str, extraction_point_name: str) -> MetricsSnapshot | None:
        """The current metrics for one binding, or None if it has none."""
        with self._dict_lock:
            stats = self._stats.get((request_id, extraction_point_name))
        if stats is None:
            return None
        with stats.lock:
            avg_latency = stats.activation_latency_total / stats.activation_count if stats.activation_count else None
            return MetricsSnapshot(
                queue_depth=stats.queue_depth,
                drop_count=stats.drop_count,
                activation_count=stats.activation_count,
                error_count=stats.error_count,
                avg_activation_latency_seconds=avg_latency,
            )


class MetricsSinkHolder:
    """Mutable single-slot box for an optionally-attached external
    `MetricsSink`, mirroring `binding.LogSinkHolder` exactly (see that
    class's docstring for why a holder rather than a constructor-time
    capture)."""

    __slots__ = ("sink",)

    def __init__(self, sink: MetricsSink | None = None) -> None:
        self.sink = sink


class MetricsRecorder:
    """Fans out every metrics event to the router's always-on
    `InMemoryMetricsRegistry` (so `Router.get_metrics` always works) and,
    if one is attached, to an external `MetricsSink` plug-in.

    This is the object every `AsyncWorker` actually holds and calls into --
    workers don't know or care whether an external sink is attached, they
    just call `MetricsRecorder`, which handles both destinations and the
    forwarding-failure isolation for the external one.
    """

    def __init__(self, registry: InMemoryMetricsRegistry, sink_holder: MetricsSinkHolder) -> None:
        self._registry = registry
        self._sink_holder = sink_holder

    def record_queue_depth(self, request_id: str, extraction_point_name: str, depth: int) -> None:
        self._registry.record_queue_depth(request_id, extraction_point_name, depth)
        self._forward("record_queue_depth", request_id, extraction_point_name, depth)

    def record_drop(self, request_id: str, extraction_point_name: str) -> None:
        self._registry.record_drop(request_id, extraction_point_name)
        self._forward("record_drop", request_id, extraction_point_name)

    def record_activation(self, request_id: str, extraction_point_name: str, latency_seconds: float) -> None:
        self._registry.record_activation(request_id, extraction_point_name, latency_seconds)
        self._forward("record_activation", request_id, extraction_point_name, latency_seconds)

    def record_probe_error(self, request_id: str, extraction_point_name: str) -> None:
        self._registry.record_probe_error(request_id, extraction_point_name)
        self._forward("record_probe_error", request_id, extraction_point_name)

    def _forward(self, method_name: str, request_id: str, extraction_point_name: str, *args: object) -> None:
        sink = self._sink_holder.sink
        if sink is None:
            return
        try:
            getattr(sink, method_name)(request_id, extraction_point_name, *args)
        except Exception:  # noqa: BLE001 -- a misbehaving metrics sink must not affect dispatch or probe processing
            logger.exception(
                "metrics sink %s.%s raised for extraction_point=%r request_id=%r",
                type(sink).__name__,
                method_name,
                extraction_point_name,
                request_id,
            )


__all__ = [
    "InMemoryMetricsRegistry",
    "MetricsRecorder",
    "MetricsSink",
    "MetricsSinkHolder",
    "MetricsSnapshot",
]

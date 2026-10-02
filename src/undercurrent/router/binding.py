"""Internal per-(request_id, extraction_point_name) state.

Not part of the public API -- `Router` is the only thing that constructs
these.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core import Probe, ProbeAction, ProbeResult, ProbeSignal
from ..spec import ActivationRecord, ExtractionPoint, InterventionPolicy
from .metrics import MetricsRecorder
from .overflow import EMPTY, BoundedDropQueue, OverflowPolicy

logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsLogSink(Protocol):
    """Structural type for whatever `Router.attach_log_sink` is handed.

    Deliberately not imported from `undercurrent.sinks` -- that package
    depends on `undercurrent.router`, not the other way around, so this file
    only describes the shape it relies on (matching
    `undercurrent.sinks.LogSink`) rather than importing it.
    """

    def write_signal(self, request_id: str, extraction_point_name: str, signal: ProbeSignal) -> None: ...

    def write_result(self, request_id: str, extraction_point_name: str, result: ProbeResult) -> None: ...


class LogSinkHolder:
    """Mutable single-slot box shared between a `Router` and every
    `AsyncWorker` it has ever created.

    Existing so `Router.attach_log_sink` can be called before or after
    `register_request` and still take effect: workers read `.sink` fresh
    on every forward rather than capturing a value at construction time.
    """

    __slots__ = ("sink",)

    def __init__(self, sink: SupportsLogSink | None = None) -> None:
        self.sink = sink


class AsyncWorker:
    """Owns one binding's bounded queue and its dedicated drain loop.

    The drain loop is submitted once to the shared thread pool and runs for
    the lifetime of the binding (see `Router`'s docstring for why threads,
    not asyncio, back the pool). It is a long-lived, single-consumer task:
    exactly one thread ever calls `probe.on_activation` for a given probe
    instance, so activations are always processed in the order `route()`
    enqueued them -- required for trajectory probes, whose state depends on
    activation order, not just activation content.

    Log sink forwarding
    --------------------
    If `log_sink_holder` carries a sink, every non-None `ProbeSignal`
    returned by `on_activation` is forwarded to it via `write_signal`, on
    this worker's own thread -- never inside `Router.route()`'s caller --
    so a slow or failing sink can only ever add latency to this one
    binding's processing, not to dispatch. `forward_result` (called by
    `Router.end_request` once this worker's queue has fully drained and
    `on_end` has returned) forwards the final `ProbeResult` via
    `write_result`; by then the worker thread has already exited, so that
    call runs on `end_request`'s caller instead -- consistent with
    `end_request` already being a blocking finalize call, unlike `route()`.
    A sink that raises is logged and swallowed either way, matching how a
    misbehaving probe is handled below.

    Failure isolation
    ------------------
    If `probe.on_activation` raises, the exception is caught here so it
    can never propagate out of this worker's loop and take down the
    shared thread pool (which every other binding, across every other
    request, also depends on). It's logged via stdlib `logging` *and*
    turned into a synthetic `ProbeSignal(action=CONTINUE, metadata={...})`
    carrying the error, which flows through the exact same log-sink
    forwarding path a real signal would -- chosen over a separate error
    channel so a caller only has to watch one place (the log sink) to see
    everything a probe reported, including its own failures, and so
    `undercurrent.sinks`'s existing sinks need no new integration to surface
    it. It's also counted via `metrics.record_probe_error`. Either way,
    the loop continues to the next queued item -- one bad activation
    never stops the rest of this binding's backlog, let alone anyone
    else's.

    Metrics
    -------
    `metrics` (a `MetricsRecorder`, always provided by `Router`) is sent
    a fresh queue-depth reading right after every enqueue and dequeue, a
    drop event whenever the underlying queue's overflow policy discards
    an item (wired through `BoundedDropQueue`'s `on_drop` callback), and
    an activation-latency reading (wall-clock time spent inside
    `on_activation`, success or failure) after every item is processed.
    """

    def __init__(
        self,
        probe: Probe,
        queue_depth: int,
        policy: OverflowPolicy,
        executor: Executor,
        metrics: MetricsRecorder,
        log_sink_holder: LogSinkHolder | None = None,
    ) -> None:
        self._probe = probe
        self._metrics = metrics
        self.queue = BoundedDropQueue(maxsize=queue_depth, policy=policy, on_drop=self._on_drop)
        self._done = threading.Event()
        self._log_sink_holder = log_sink_holder
        executor.submit(self._run)

    def submit(self, record: ActivationRecord) -> None:
        self.queue.put(record)
        self._record_queue_depth()

    def _on_drop(self) -> None:
        self._metrics.record_drop(self._probe.request_id, self._probe.extraction_point_name)

    def _record_queue_depth(self) -> None:
        self._metrics.record_queue_depth(self._probe.request_id, self._probe.extraction_point_name, len(self.queue))

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get(timeout=None)
                if item is EMPTY:
                    break
                self._record_queue_depth()
                signal: ProbeSignal | None
                start = time.monotonic()
                try:
                    signal = self._probe.on_activation(item)
                except Exception as exc:  # noqa: BLE001 -- one misbehaving probe must not take down the worker pool
                    logger.exception(
                        "probe %s.on_activation raised for extraction_point=%r request_id=%r",
                        type(self._probe).__name__,
                        self._probe.extraction_point_name,
                        self._probe.request_id,
                    )
                    self._metrics.record_probe_error(self._probe.request_id, self._probe.extraction_point_name)
                    signal = ProbeSignal(
                        action=ProbeAction.CONTINUE,
                        metadata={
                            "router_error": True,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "extraction_point_name": self._probe.extraction_point_name,
                        },
                    )
                finally:
                    self._metrics.record_activation(
                        self._probe.request_id, self._probe.extraction_point_name, time.monotonic() - start
                    )
                if signal is not None:
                    self._forward("write_signal", signal)
        finally:
            self._done.set()

    def forward_result(self, result: ProbeResult) -> None:
        """Called by `Router.end_request` after `probe.on_end` returns."""
        self._forward("write_result", result)

    def _forward(self, method_name: str, payload: Any) -> None:
        holder = self._log_sink_holder
        sink = holder.sink if holder is not None else None
        if sink is None:
            return
        try:
            getattr(sink, method_name)(self._probe.request_id, self._probe.extraction_point_name, payload)
        except Exception:  # noqa: BLE001 -- a misbehaving log sink must not affect dispatch or probe processing
            logger.exception(
                "log sink %s.%s raised for extraction_point=%r request_id=%r",
                type(sink).__name__,
                method_name,
                self._probe.extraction_point_name,
                self._probe.request_id,
            )

    def stop_and_drain(self, timeout: float | None) -> None:
        """Signal that no more items will arrive, and block until every
        item queued as of now has been processed (or `timeout` elapses)."""
        self.queue.close()
        if not self._done.wait(timeout=timeout):
            logger.warning(
                "async worker for extraction_point=%r request_id=%r did not finish draining within %ss",
                self._probe.extraction_point_name,
                self._probe.request_id,
                timeout,
            )

    def cancel(self) -> None:
        """Signal that no more items will arrive, and discard whatever is
        currently queued without processing it -- used by
        `Router.shutdown(wait=False)` for an immediate, non-draining
        teardown. Does not wait for `_run` to exit.

        Known limitation, inherent to using real OS threads with no
        cooperative cancellation protocol: if this binding's worker is
        already inside `probe.on_activation` for some item when `cancel`
        is called, that one call still runs to completion -- only the
        backlog queued *behind* it is discarded. There is no way to
        preempt a plain Python thread mid-call short of killing the
        process, and a real probe's `on_activation` isn't expected to poll
        for cancellation.
        """
        self.queue.close_and_discard()


@dataclass
class Binding:
    """Everything the router needs for one (request_id, extraction_point_name) pair."""

    extraction_point: ExtractionPoint
    probe: Probe
    worker: AsyncWorker | None = None  # None for inline; set for async
    # The *effective* InterventionPolicy for this binding -- point-level
    # override if it had one, else the Router's default_intervention_policy.
    # Resolved once at register_request() time (see Router._effective_intervention),
    # not re-read from the ExtractionPoint on every dispatch.
    intervention: InterventionPolicy = field(default_factory=InterventionPolicy)

    def dispatch(self, record: ActivationRecord) -> ProbeSignal | None:
        """Plain, un-timed dispatch: used for async bindings (queued,
        returns None immediately) and for inline bindings whose effective
        intervention is REJECT (call and return whatever on_activation
        produces, however long that naturally takes -- no wait-for-decision
        contract). `Router.route()` calls this directly in both those
        cases, and instead calls `Router._dispatch_with_intervention()` for
        an inline binding whose effective mode is BLOCK_UNTIL_SIGNAL (that
        path needs the router's shared executor and circuit-breaker
        bookkeeping, which don't belong on this lightweight dataclass)."""
        if self.worker is None:
            return self.probe.on_activation(record)
        self.worker.submit(record)
        return None

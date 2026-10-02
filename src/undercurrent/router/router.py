"""Router: dispatches ActivationRecords to the probe instance(s) registered
for their request, according to each extraction point's execution_mode.

Worker model
------------
A single shared `concurrent.futures.ThreadPoolExecutor` backs every async
extraction point across every request. Threads, not asyncio: undercurrent.core's
`Probe.on_activation` is a plain synchronous method, and there is no
expectation it won't itself block (a real MLP forward pass, a GPU call,
...). Running arbitrary blocking probe code inside an asyncio event loop
would either stall the loop or require wrapping every call in
`run_in_executor` -- which is a thread pool with extra steps. Using threads
directly also keeps the router synchronous end-to-end, so it can be driven
from ordinary adapter/test code with no event loop required.

Each async binding gets one dedicated long-lived task on that pool (see
`binding.AsyncWorker`): a single-consumer drain loop that owns the
binding's queue for the binding's whole lifetime. This is what guarantees
in-order processing for trajectory probes (whose state depends on
activation order), at the cost of a known limitation worth stating
plainly: if more than `worker_pool_size` async bindings are alive at once
across all requests, the extra ones queue for a worker and receive no
service (their own bounded queues just apply their overflow policy) until
an earlier binding's request ends and frees a slot. For this package's
scope (no real engine adapter, unit-tested via synthetic streams) that
tradeoff is fine; a production deployment with high async-probe
concurrency would want a fair scheduler that time-slices many bindings
over a smaller thread count instead of dedicating one thread each.

Isolation
---------
All per-request state lives nested under `request_id` in a single dict
(`Router._requests`); nothing else in this class is keyed or aggregated
across request_ids (see `tests/test_isolation.py`). Removing a `request_id`
key in `end_request` tears down everything for that request in one step.

Observation logging (`attach_log_sink`)
----------------------------------------
`Router.attach_log_sink(sink)` is the hook a caller uses to wire an
out-of-band observation log (e.g. `undercurrent.sinks.FileLogSink` /
`WebhookLogSink`) onto every async ("observe mode") binding: each
`ProbeSignal` an async trajectory probe emits is forwarded to
`sink.write_signal(...)` from the binding's own worker thread, and its
final `ProbeResult` is forwarded to `sink.write_result(...)` once
`end_request` finalizes it. `sink` only needs to duck-type
`binding.SupportsLogSink` (`write_signal`/`write_result`) -- this package
takes no dependency on `undercurrent.sinks`. See `Router.attach_log_sink`'s
docstring for the non-blocking guarantee.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from ..core import Probe, ProbeAction, ProbeFactory, ProbeResult, ProbeSignal, RequestContext
from ..core.registry import ProbeNotFoundError, ProbeRegistry, default_registry
from ..errors import ProbingValueError, did_you_mean
from ..spec import (
    ActivationRecord,
    ExecutionMode,
    ExtractionPoint,
    InterventionMode,
    InterventionPolicy,
    ProbeKind,
    ProbeSpec,
    TimeoutAction,
)
from .binding import AsyncWorker, Binding, LogSinkHolder, SupportsLogSink
from .errors import RouterError
from .metrics import InMemoryMetricsRegistry, MetricsRecorder, MetricsSink, MetricsSinkHolder, MetricsSnapshot
from .overflow import OverflowPolicy
from .request import RequestHandle

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_DEPTH = 32
"""Default ``Router(default_queue_depth=...)``: queue depth for async points that don't set ``queue_depth``."""
DEFAULT_DRAIN_TIMEOUT = 30.0
"""Default ``Router(drain_timeout=...)``, in seconds."""
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
"""Default ``Router(circuit_breaker_threshold=...)``."""

RequestEndListener = Callable[[str, "dict[str, ProbeResult]"], None]
"""``listener(request_id, results)``, registered with [`Router.on_request_end`][undercurrent.router.Router.on_request_end]."""

_SHUT_DOWN_MESSAGE = "router has been shut down; create a new Router"


_NOT_REGISTERED_HINT = (
    "Register it first with router.register_request(...) or `with router.request(...)`, "
    "and use it only until end_request()."
)


def _no_such_point(where: str, name: str, request_id: str, bindings: Mapping[str, Any]) -> str:
    known = sorted(bindings)
    return (
        f"{where}: no extraction point named {name!r} is registered for request_id={request_id!r}."
        f"{did_you_mean(name, known)} Registered: {', '.join(repr(k) for k in known) or '(none)'}."
    )


def default_worker_pool_size() -> int:
    """The default ``worker_pool_size``: ``min(32, os.cpu_count() * 4)``.

    Pool threads mostly wait (on their binding's queue, or inside a probe
    blocked on I/O or a GPU call), so the pool is oversubscribed relative to
    the core count. Each async binding holds one thread for its request's
    whole lifetime, so this number is how many async bindings are serviced
    at once. If you expect more concurrently live async bindings, pass
    ``Router(worker_pool_size=...)`` explicitly.
    """
    # Why "* 4" rather than ThreadPoolExecutor's "+ 4": every thread is always a
    # waiter of one kind or another (queue.get, or a probe blocked on I/O / an
    # accelerator call), never tight CPU-bound Python, so oversubscribing well past
    # the core count is safe and lets more bindings' blocking calls overlap.
    # min(32, ...) caps it: past a couple dozen threads the OS-thread overhead
    # starts to matter, and since each async binding pins one thread, growing this
    # without bound doesn't scale request concurrency anyway.
    return min(32, (os.cpu_count() or 1) * 4)


def _as_probe_registry(
    probe_registry: Mapping[str, ProbeFactory | type[Probe]] | ProbeRegistry | None,
) -> ProbeRegistry:
    if probe_registry is None:
        return default_registry
    if isinstance(probe_registry, ProbeRegistry):
        return probe_registry
    registry = ProbeRegistry(load_entry_points=False)
    for name, probe in probe_registry.items():
        registry.register(name, probe)
    return registry


class Router:
    """Dispatches each [`ActivationRecord`][undercurrent.spec.ActivationRecord] to the probe instances of its request.

    Inline extraction points run on the caller's thread and can return an
    ``ABORT`` signal; async ones run on a shared, bounded worker pool and
    only observe. Every request gets its own freshly spawned probe instances,
    and nothing is shared across requests.

    ```python
    with Router() as router:  # probes come from @register_probe
        with router.request(extraction_points=spec) as req:
            for record in activation_stream:
                signal = req.route(record)  # non-None only for inline points
                if signal is not None and signal.action is ProbeAction.ABORT:
                    break
        results = req.results  # {extraction_point_name: ProbeResult}
    ```

    ``router.request(...)`` wraps the lower-level
    [`register_request`][undercurrent.router.Router.register_request] /
    [`route`][undercurrent.router.Router.route] /
    [`end_request`][undercurrent.router.Router.end_request] calls. Call
    [`shutdown`][undercurrent.router.Router.shutdown] (or use
    ``with Router(...)``) when done.

    Each async extraction point of a live request holds one worker thread
    until its request ends, which keeps its activations in order. If more
    than ``worker_pool_size`` async bindings are alive at once, the extra
    ones wait for a free worker (their bounded queues apply their overflow
    policy meanwhile).
    """

    def __init__(
        self,
        probe_registry: Mapping[str, ProbeFactory | type[Probe]] | ProbeRegistry | None = None,
        *,
        worker_pool_size: int | None = None,
        default_queue_depth: int = DEFAULT_QUEUE_DEPTH,
        default_overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        drain_timeout: float = DEFAULT_DRAIN_TIMEOUT,
        metrics_sink: MetricsSink | None = None,
        default_intervention_policy: InterventionPolicy | None = None,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    ) -> None:
        """
        Args:
            probe_registry: where ``probe_type`` names are looked up. ``None``
                (the default) uses the global registry that ``@register_probe``
                fills (plus ``undercurrent.probes`` entry-point plugins),
                consulted at ``register_request`` time, so probes registered
                after the router is built still resolve. A
                [`ProbeRegistry`][undercurrent.core.ProbeRegistry] is used as-is.
                A mapping of ``probe_type`` -> ``ProbeFactory`` or ``Probe``
                subclass is copied and is authoritative: no fallback to the
                global registry or to plugins.
            worker_pool_size: number of threads backing all async bindings,
                shared across every request. Defaults to
                [`default_worker_pool_size()`][undercurrent.router.default_worker_pool_size].
            default_queue_depth: queue depth for an async extraction point whose
                ``queue_depth`` is None.
            default_overflow_policy: what a full async queue does; see
                [`OverflowPolicy`][undercurrent.router.OverflowPolicy].
            drain_timeout: seconds ``end_request`` waits for an async binding's
                queue to drain before finalizing anyway (logged, not raised).
            metrics_sink: an optional external
                [`MetricsSink`][undercurrent.router.MetricsSink], the same as
                calling ``attach_metrics_sink`` right after construction.
                ``get_metrics`` works with or without one.
            default_intervention_policy: policy for an extraction point whose
                ``intervention`` is None. Defaults to ``InterventionPolicy()``
                (``mode=reject``, no waiting).
            circuit_breaker_threshold: number of consecutive timeouts or
                exceptions (across requests) after which a ``block_until_signal``
                extraction point is permanently downgraded to ``mode=reject`` by
                this router.

        Raises:
            ProbingValueError: ``worker_pool_size < 1``.
        """
        # Maintainer notes on the arguments:
        # - default_overflow_policy is used unless an extraction point carries its own
        #   overflow_policy attribute; the spec schema doesn't define one today, so
        #   it is read defensively with getattr.
        # - default_intervention_policy mirrors default_queue_depth: an extraction
        #   point's intervention=None means "not specified here".
        # - circuit_breaker_threshold: see route() and _trip_circuit_breaker.
        self._probe_registry = _as_probe_registry(probe_registry)
        self._default_queue_depth = default_queue_depth
        self._default_overflow_policy = default_overflow_policy
        self._drain_timeout = drain_timeout
        self.worker_pool_size = worker_pool_size if worker_pool_size is not None else default_worker_pool_size()
        if isinstance(self.worker_pool_size, int) and self.worker_pool_size < 1:
            # ThreadPoolExecutor would raise a bare ValueError for this anyway.
            raise ProbingValueError(
                f"Router(worker_pool_size={self.worker_pool_size!r}): worker_pool_size must be >= 1. "
                "Omit it to use default_worker_pool_size()."
            )
        self._executor = ThreadPoolExecutor(max_workers=self.worker_pool_size, thread_name_prefix="probing-router")

        self._requests: dict[str, dict[str, Binding]] = {}
        self._request_contexts: dict[str, RequestContext] = {}
        self._lock = threading.RLock()
        self._log_sink_holder = LogSinkHolder()
        self._metrics_registry = InMemoryMetricsRegistry()
        self._metrics_sink_holder = MetricsSinkHolder(metrics_sink)
        self._metrics_recorder = MetricsRecorder(self._metrics_registry, self._metrics_sink_holder)

        self._default_intervention_policy = (
            default_intervention_policy if default_intervention_policy is not None else InterventionPolicy()
        )
        self._circuit_breaker_threshold = circuit_breaker_threshold
        # Keyed by extraction_point_name, deliberately NOT nested under
        # request_id like every other piece of state this class tracks --
        # the circuit breaker's whole point is to persist across different
        # requests that reuse the same extraction point name (see
        # `route()`'s module-level docstring reference and
        # `_trip_circuit_breaker`). Never torn down by `end_request`.
        self._intervention_failure_counts: dict[str, int] = {}
        self._intervention_tripped: set[str] = set()

        # See `on_request_end`. Guarded by its own lock so registering or
        # removing a listener never contends with dispatch on `self._lock`.
        self._request_end_listeners: list[tuple[object, RequestEndListener]] = []
        self._listeners_lock = threading.Lock()
        self._closed = False

    def attach_log_sink(self, sink: SupportsLogSink | None) -> None:
        """Forward async extraction points' signals and results to ``sink``.

        ``sink`` is a [`LogSink`][undercurrent.sinks.LogSink] (anything with
        ``write_signal`` and ``write_result``). Every async binding calls
        ``sink.write_signal(...)`` for each non-None signal its probe returns,
        and ``sink.write_result(...)`` once ``end_request`` finalizes it. Inline
        extraction points don't forward: their signals already go to
        ``route()``'s caller.

        Signals are forwarded from the binding's own worker thread, never from
        ``route()``'s caller, so a slow or raising sink can't add latency to
        dispatch or affect other bindings. Results are forwarded inside
        ``end_request``, after the binding's queue has drained.

        Takes effect immediately for every binding, present and future, so it
        may be called before or after ``register_request``. Pass ``None`` to
        detach.
        """
        # Bindings read the sink through a shared LogSinkHolder rather than capturing
        # it at construction; see binding.AsyncWorker for the forwarding detail.
        self._log_sink_holder.sink = sink

    def attach_metrics_sink(self, sink: MetricsSink | None) -> None:
        """Forward every async-binding metrics event to an external [`MetricsSink`][undercurrent.router.MetricsSink].

        Use it for a Prometheus or StatsD backend. Events (queue depth, drops,
        activation latency, probe errors) still go to the router's own in-memory
        registry that ``get_metrics`` reads; this adds a second destination.

        Like ``attach_log_sink``: events are forwarded from each binding's worker
        thread, never ``route()``'s caller; a raising sink is caught, logged and
        ignored; it takes effect immediately for every binding. Pass ``None`` to
        detach.
        """
        self._metrics_sink_holder.sink = sink

    def get_metrics(self, request_id: str, extraction_point_name: str) -> MetricsSnapshot:
        """Return the current [`MetricsSnapshot`][undercurrent.router.MetricsSnapshot] for one async binding.

        Only valid while ``request_id`` is registered: ``end_request`` tears the
        metrics down with everything else for that request.

        Raises:
            RouterError: ``request_id`` isn't registered (or has ended),
                ``extraction_point_name`` is unknown, or the extraction point is
                inline (nothing is queued or measured for those).
        """
        with self._lock:
            bindings = self._requests.get(request_id)
            if bindings is None:
                raise RouterError(f"get_metrics(): request_id={request_id!r} is not registered. {_NOT_REGISTERED_HINT}")
            binding = bindings.get(extraction_point_name)
            if binding is None:
                raise RouterError(_no_such_point("get_metrics()", extraction_point_name, request_id, bindings))
            if binding.worker is None:
                raise RouterError(
                    f"get_metrics(): extraction point {extraction_point_name!r} is inline, so it has no queue "
                    "metrics. Metrics exist only for execution_mode='async' points."
                )
        snapshot = self._metrics_registry.snapshot(request_id, extraction_point_name)
        if snapshot is None:
            # register_request always seeds a zeroed entry, so this
            # shouldn't happen -- degrade gracefully rather than raising.
            snapshot = MetricsSnapshot(
                queue_depth=0, drop_count=0, activation_count=0, error_count=0, avg_activation_latency_seconds=None
            )
        return snapshot

    def register_request(
        self,
        request_id: str,
        extraction_points: ProbeSpec | Iterable[ExtractionPoint],
        request_ctx: RequestContext,
    ) -> None:
        """Spawn a fresh probe instance for each extraction point and start it.

        All-or-nothing: every extraction point is validated before any probe is
        spawned, so an invalid one never leaves the request partially
        registered. Prefer [`request`][undercurrent.router.Router.request], which
        also guarantees ``end_request`` runs.

        Args:
            request_id: unique among in-flight requests.
            extraction_points: a ``ProbeSpec`` or any iterable of ``ExtractionPoint``.
            request_ctx: passed to every probe's ``on_start`` and ``on_end``.

        Raises:
            RouterError: the router is shut down, ``request_id`` is already
                registered, a ``probe_type`` can't be resolved, or an extraction
                point is invalid for the router.
        """
        extraction_points = list(extraction_points)
        with self._lock:
            if self._closed:
                raise RouterError(_SHUT_DOWN_MESSAGE)
            if request_id in self._requests:
                raise RouterError(
                    f"register_request(): request_id {request_id!r} is already registered. Request ids must be "
                    "unique among in-flight requests: end the earlier one with end_request(), or use a fresh id "
                    "(e.g. str(uuid.uuid4()))."
                )

            factories: dict[str, ProbeFactory] = {}
            for point in extraction_points:
                self._validate_extraction_point(point)
                factories[point.name] = self._resolve_probe_factory(point)

            bindings: dict[str, Binding] = {}
            for point in extraction_points:
                probe_factory = factories[point.name]
                # The point's own probe_args override the factory's kwargs.
                # getattr: an ExtractionPoint-like object may predate probe_args.
                probe = probe_factory.spawn(request_id, point.name, **getattr(point, "probe_args", {}))

                worker: AsyncWorker | None = None
                if point.execution_mode == ExecutionMode.ASYNC:
                    queue_depth = point.queue_depth or self._default_queue_depth
                    policy = getattr(point, "overflow_policy", None) or self._default_overflow_policy
                    worker = AsyncWorker(
                        probe,
                        queue_depth,
                        policy,
                        self._executor,
                        self._metrics_recorder,
                        log_sink_holder=self._log_sink_holder,
                    )
                    self._metrics_registry.register(request_id, point.name)

                bindings[point.name] = Binding(
                    extraction_point=point,
                    probe=probe,
                    worker=worker,
                    intervention=self._effective_intervention(point),
                )

            for binding in bindings.values():
                binding.probe.on_start(request_ctx)

            self._requests[request_id] = bindings
            self._request_contexts[request_id] = request_ctx

    def _effective_intervention(self, point: ExtractionPoint) -> InterventionPolicy:
        """The InterventionPolicy that actually governs dispatch for `point`:
        its own `intervention` if it set one, else this router's
        `default_intervention_policy` -- mirrors `default_queue_depth`'s
        fallback exactly. Read via `getattr` (not direct attribute access)
        so a hand-built `ExtractionPoint` predating this field (or a stub
        used in isolated testing) degrades to "unspecified" rather than
        raising `AttributeError`.

        A circuit-breaker trip (see `_trip_circuit_breaker`) is NOT folded
        in here -- it's applied at dispatch time in `route()` instead, so
        that a still-registered request whose binding was created with
        `block_until_signal` *before* the breaker tripped is immediately
        protected too, not just requests registered afterward.
        """
        policy = getattr(point, "intervention", None)
        return policy if policy is not None else self._default_intervention_policy

    def _resolve_probe_factory(self, point: ExtractionPoint) -> ProbeFactory:
        """Look up `point.probe_type` and check the probe class implements
        the `probe_kind` the point declares -- a mismatch (e.g. a trajectory
        probe wired to a single_shot point) is a spec wiring bug better
        caught here than partway through a request."""
        try:
            factory = self._probe_registry.get(point.probe_type)
        except ProbeNotFoundError as exc:
            raise RouterError(f"extraction point {point.name!r}: {exc}") from exc

        declared = getattr(point, "probe_kind", None)
        if declared is not None and ProbeKind(declared) != ProbeKind(factory.probe_cls.probe_kind):
            raise RouterError(
                f"extraction point {point.name!r}: probe_type={point.probe_type!r} is "
                f"{factory.probe_cls.__qualname__}, which implements probe_kind="
                f"{factory.probe_cls.probe_kind!r}, but the spec declares probe_kind={ProbeKind(declared).value!r}. "
                f"Set probe_kind: {factory.probe_cls.probe_kind} on this extraction point, or use a probe_type "
                f"that implements {ProbeKind(declared).value!r}."
            )
        return factory

    def _validate_extraction_point(self, point: ExtractionPoint) -> None:
        # undercurrent.spec's parser should already reject this combination at
        # spec-parse time -- but the router must not trust that every
        # ExtractionPoint it's handed necessarily went through that path
        # (hand-built in a test, constructed by some future caller that
        # bypasses the parser, ...), so it re-checks defensively here and
        # fails loudly rather than silently mis-scheduling an async probe
        # that has no defined semantics.
        if point.execution_mode == ExecutionMode.ASYNC and point.probe_kind == ProbeKind.SINGLE_SHOT:
            raise RouterError(
                f"extraction point {point.name!r}: execution_mode=async is not valid with probe_kind=single_shot. "
                "Use execution_mode=inline, or a trajectory probe with probe_kind=trajectory."
            )

        # Same defensive posture as above, for the newer rule: async is
        # already fire-and-forget (route() always returns None for it,
        # immediately -- see `route()`), so a block_until_signal policy
        # attached to one (whether set directly on the point or inherited
        # from this router's own default_intervention_policy) has no
        # defined semantics. undercurrent.spec.schema also rejects this
        # combination at parse time for anything built via parse_dict/
        # parse_yaml (see ExtractionPointSpec's model validator) -- this is
        # the same "don't trust the caller went through the parser" belt
        # that the check above applies.
        effective = self._effective_intervention(point)
        if point.execution_mode == ExecutionMode.ASYNC and effective.mode != InterventionMode.REJECT:
            raise RouterError(
                f"extraction point {point.name!r}: execution_mode=async cannot use an "
                f"intervention policy other than the default (mode=reject) -- async cannot "
                f"participate in synchronous intervention (effective mode={effective.mode.value!r}). "
                "Use execution_mode=inline for this point, or give it intervention mode=reject "
                "(also check Router(default_intervention_policy=...))."
            )

    def route(self, record: ActivationRecord) -> ProbeSignal | None:
        """Dispatch one activation to the probe registered for its (request_id, extraction_point_name).

        - **Inline, default policy** (``mode=reject``): calls the probe's
          ``on_activation`` on this thread and returns its
          [`ProbeSignal`][undercurrent.core.ProbeSignal] (or None). An engine
          adapter uses this to decide whether to abort generation.
        - **Inline, ``block_until_signal``**: waits up to ``timeout_ms`` for the
          same call and returns the ``on_timeout`` fallback signal if it doesn't
          finish in time or raises. Once the extraction point's circuit breaker
          has tripped (see ``circuit_breaker_threshold``), it is dispatched as
          plain ``reject`` instead, permanently.
        - **Async**: pushes the record onto the binding's queue (subject to its
          overflow policy) and returns None immediately.

        Raises:
            RouterError: the request isn't registered, the extraction point is
                unknown, or the record doesn't match its extraction point.
        """
        # block_until_signal dispatch: _dispatch_with_intervention. Circuit breaker:
        # _trip_circuit_breaker.
        with self._lock:
            bindings = self._requests.get(record.request_id)
            if bindings is None:
                if self._closed:
                    raise RouterError(_SHUT_DOWN_MESSAGE)
                raise RouterError(
                    f"route(): request_id={record.request_id!r} is not registered or has already ended. "
                    f"{_NOT_REGISTERED_HINT}"
                )
            binding = bindings.get(record.extraction_point_name)

        if binding is None:
            raise RouterError(_no_such_point("route()", record.extraction_point_name, record.request_id, bindings))

        self._validate_record_matches_point(record, binding.extraction_point)

        if (
            binding.worker is None
            and binding.intervention.mode == InterventionMode.BLOCK_UNTIL_SIGNAL
            and binding.extraction_point.name not in self._intervention_tripped
        ):
            return self._dispatch_with_intervention(binding, record)
        return binding.dispatch(record)

    def _dispatch_with_intervention(self, binding: Binding, record: ActivationRecord) -> ProbeSignal | None:
        """`block_until_signal` dispatch: run `on_activation` on the shared
        executor and wait up to `binding.intervention.timeout_ms` for it.

        Known limitation, inherent to using real OS threads with no
        cooperative cancellation protocol (same caveat `AsyncWorker.cancel`
        documents for the same reason): if `on_activation` is still running
        when `timeout_ms` elapses, this method returns the `on_timeout`
        fallback signal immediately, but the original call is NOT
        preempted -- it keeps running in the background and its eventual
        return value (or exception) is simply discarded when it finishes.
        For a `trajectory` probe with mutable per-call state, this means a
        second `on_activation` call dispatched for the same probe instance
        before the first one actually finishes running (e.g. the very next
        token, if generation wasn't told to stop) executes concurrently
        with it -- a real data race for any probe that isn't itself
        thread-safe. This is a deliberate, documented tradeoff (matching
        this codebase's existing style of stating a real limitation rather
        than pretending to solve it): serializing calls to wait out a
        straggler would turn one slow activation into unbounded backlog for
        every later one, which is a worse failure mode than an occasional
        overlap. A probe author whose probe will run under
        `block_until_signal` should either keep `on_activation` cheap
        relative to `timeout_ms`, or make its own state access safe against
        this specific overlap.
        """
        policy = binding.intervention
        # InterventionPolicy.__post_init__ rejects block_until_signal without a timeout.
        assert policy.timeout_ms is not None
        extraction_point_name = binding.extraction_point.name
        future = self._executor.submit(binding.probe.on_activation, record)
        try:
            signal = future.result(timeout=policy.timeout_ms / 1000.0)
        except FutureTimeoutError:
            self._record_intervention_outcome(record.request_id, extraction_point_name, failed=True, reason="timeout")
            return self._fallback_signal(policy, extraction_point_name, reason="timeout")
        except Exception as exc:  # noqa: BLE001 -- mirrors AsyncWorker's isolation of a misbehaving probe
            logger.exception(
                "probe %s.on_activation raised during block_until_signal dispatch for "
                "extraction_point=%r request_id=%r",
                type(binding.probe).__name__,
                extraction_point_name,
                record.request_id,
            )
            self._record_intervention_outcome(record.request_id, extraction_point_name, failed=True, reason="exception")
            return self._fallback_signal(policy, extraction_point_name, reason="exception", error=exc)
        else:
            self._record_intervention_outcome(record.request_id, extraction_point_name, failed=False)
            return signal

    @staticmethod
    def _fallback_signal(
        policy: InterventionPolicy,
        extraction_point_name: str,
        reason: str,
        error: BaseException | None = None,
    ) -> ProbeSignal:
        action = ProbeAction.ABORT if policy.on_timeout == TimeoutAction.ABORT else ProbeAction.CONTINUE
        metadata: dict[str, Any] = {
            "intervention_fallback": True,
            "reason": reason,
            "extraction_point_name": extraction_point_name,
            "timeout_ms": policy.timeout_ms,
        }
        if error is not None:
            metadata["error_type"] = type(error).__name__
            metadata["error"] = str(error)
        return ProbeSignal(action=action, metadata=metadata)

    def _record_intervention_outcome(
        self, request_id: str, extraction_point_name: str, *, failed: bool, reason: str | None = None
    ) -> None:
        """Circuit-breaker bookkeeping for `block_until_signal` dispatch.

        A success resets the consecutive-failure count for this extraction
        point to zero. A failure (timeout or exception) increments it, and
        once it reaches `circuit_breaker_threshold`, trips the breaker --
        counted per extraction_point_name, deliberately across whichever
        different request_ids happen to share that name (see `__init__`'s
        docstring and the module docstring's isolation note: this is the
        one piece of state in this class that intentionally is NOT scoped
        to a single request_id, because the whole point is to notice a
        probe that is systematically unsafe across many requests, not just
        unlucky once).
        """
        tripped_now = False
        failure_count = 0
        with self._lock:
            if failed:
                failure_count = self._intervention_failure_counts.get(extraction_point_name, 0) + 1
                self._intervention_failure_counts[extraction_point_name] = failure_count
                if (
                    failure_count >= self._circuit_breaker_threshold
                    and extraction_point_name not in self._intervention_tripped
                ):
                    self._intervention_tripped.add(extraction_point_name)
                    tripped_now = True
            else:
                self._intervention_failure_counts[extraction_point_name] = 0

        if tripped_now:
            self._trip_circuit_breaker(request_id, extraction_point_name, failure_count, reason)

    def _trip_circuit_breaker(
        self, request_id: str, extraction_point_name: str, failure_count: int, reason: str | None
    ) -> None:
        """Called exactly once per extraction point, the moment its
        consecutive `block_until_signal` failure count reaches
        `circuit_breaker_threshold`. From this point on, `route()` treats
        every future dispatch for this extraction_point_name (this
        request's remaining activations, and every later request that
        reuses this name) as plain reject dispatch, regardless of its
        configured `InterventionPolicy` -- see the check in `route()`.

        Logs loudly via stdlib `logging` (visible with no sink configured
        at all) and, if a log sink is attached, forwards a synthetic
        `ProbeSignal(action=FLAG, ...)` through it via `write_signal` --
        the same "reuse the existing log-sink plumbing for an out-of-band
        event, tagged in metadata" pattern `AsyncWorker` already uses to
        surface a probe's own raised exceptions. A raising sink is caught
        and logged, never allowed to affect dispatch, matching every other
        log-sink call site in this package.
        """
        logger.warning(
            "undercurrent.router: extraction_point=%r tripped its intervention circuit breaker after "
            "%d consecutive %s during block_until_signal dispatch -- downgrading to observe-only "
            "(mode=reject) for every future dispatch of this extraction point",
            extraction_point_name,
            failure_count,
            reason or "failures",
        )
        signal = ProbeSignal(
            action=ProbeAction.FLAG,
            metadata={
                "circuit_breaker_tripped": True,
                "extraction_point_name": extraction_point_name,
                "consecutive_failures": failure_count,
                "reason": reason,
            },
        )
        sink = self._log_sink_holder.sink
        if sink is None:
            return
        try:
            sink.write_signal(request_id, extraction_point_name, signal)
        except Exception:  # noqa: BLE001 -- a misbehaving log sink must not affect dispatch
            logger.exception(
                "log sink %s.write_signal raised while reporting a circuit breaker trip for extraction_point=%r",
                type(sink).__name__,
                extraction_point_name,
            )

    @staticmethod
    def _validate_record_matches_point(record: ActivationRecord, point: ExtractionPoint) -> None:
        # record.extraction_point_name is the actual routing key -- it's
        # what an ActivationRecord means to name the extraction point that
        # produced it. Re-deriving a position match here would need
        # prompt_len, which ActivationRecord doesn't carry, and would
        # duplicate the adapter's job (the adapter is what decided this
        # activation matched a position selector in the first place). What
        # the router can and does check defensively is that the record is
        # at least internally consistent with the point it claims to
        # belong to, catching a mistagged/corrupt record early and loudly.
        if record.layer not in point.layers:
            raise RouterError(
                f"record for extraction point {point.name!r} has layer={record.layer}, "
                f"not in its configured layers {point.layers}. The adapter (or code building "
                "ActivationRecords) tagged this record with the wrong layer or extraction point."
            )
        point_tensor_type = getattr(point.tensor_type, "value", point.tensor_type)
        if record.tensor_type != point_tensor_type:
            raise RouterError(
                f"record for extraction point {point.name!r} has tensor_type={record.tensor_type!r}, "
                f"expected {point_tensor_type!r}. The adapter (or code building ActivationRecords) "
                "tagged this record with the wrong tensor_type or extraction point."
            )

    def end_request(self, request_id: str) -> dict[str, ProbeResult]:
        """Finalize every probe registered for ``request_id`` and tear down its state.

        Each async binding's queue is closed and drained (backlogged activations
        are processed, up to ``drain_timeout``) before ``on_end`` is called, so a
        trajectory probe's verdict reflects everything it saw, including for a
        request that was aborted mid-stream. Then every
        [`on_request_end`][undercurrent.router.Router.on_request_end] listener is
        called with ``(request_id, results)``.

        Returns:
            ``{extraction_point_name: ProbeResult}``.

        Raises:
            RouterError: ``request_id`` isn't registered or has already ended.
        """
        # The request_id is removed from _requests before any draining happens, so a
        # concurrent route() for it fails loudly instead of racing the teardown.
        with self._lock:
            bindings = self._requests.pop(request_id, None)
            request_ctx = self._request_contexts.pop(request_id, None)

        if bindings is None or request_ctx is None:
            if self._closed:
                raise RouterError(_SHUT_DOWN_MESSAGE)
            raise RouterError(
                f"end_request(): request_id={request_id!r} is not registered or has already ended. "
                "Call end_request() once per registered request (`with router.request(...)` does it for you)."
            )

        for binding in bindings.values():
            if binding.worker is not None:
                binding.worker.stop_and_drain(timeout=self._drain_timeout)

        results: dict[str, ProbeResult] = {}
        for name, binding in bindings.items():
            result = binding.probe.on_end(request_ctx)
            results[name] = result
            if binding.worker is not None:
                binding.worker.forward_result(result)
                # Metrics are per-request state like everything else this
                # class tracks -- torn down here rather than left to grow
                # unbounded across the router's lifetime. get_metrics is
                # only ever valid while request_id is still registered,
                # same rule as get_probe.
                self._metrics_registry.forget(request_id, name)

        self._notify_request_end(request_id, results)
        return results

    def request(
        self,
        extraction_points: ProbeSpec | Iterable[ExtractionPoint],
        *,
        request_id: str | None = None,
        prompt_metadata: Mapping[str, Any] | None = None,
    ) -> RequestHandle:
        """Context manager for one request: registers it on entry and always calls ``end_request`` on exit.

        ```python
        with router.request(extraction_points=spec, prompt_metadata={"prompt": p}) as req:
            for record in stream:
                sig = req.route(record)
        req.results   # {extraction_point_name: ProbeResult}
        ```

        ``end_request`` runs even when the body raises; see
        [`RequestHandle`][undercurrent.router.RequestHandle] for how errors are
        reported.

        Args:
            extraction_points: a ``ProbeSpec`` or any iterable of ``ExtractionPoint``.
            request_id: defaults to a fresh ``uuid4().hex``.
            prompt_metadata: becomes ``RequestContext.prompt_metadata``
                (an empty dict when omitted).
        """
        return RequestHandle(
            self,
            request_id if request_id is not None else uuid.uuid4().hex,
            list(extraction_points),
            prompt_metadata,
        )

    def on_request_end(self, listener: RequestEndListener) -> Callable[[], None]:
        """Call ``listener(request_id, results)`` at the end of every ``end_request``.

        This is how a caller that doesn't call ``end_request`` itself (an engine
        adapter does it internally) still gets each request's results.

        Listeners run on ``end_request``'s calling thread, outside the router's
        internal lock (so a listener may call back into the router), in
        registration order. Each gets its own shallow copy of ``results``. A
        raising listener is logged and never affects ``end_request`` or other
        listeners.

        Returns:
            A callable that removes this listener; calling it more than once is
                a no-op.
        """
        token = object()
        with self._listeners_lock:
            self._request_end_listeners.append((token, listener))

        def remove() -> None:
            with self._listeners_lock:
                self._request_end_listeners = [entry for entry in self._request_end_listeners if entry[0] is not token]

        return remove

    def _notify_request_end(self, request_id: str, results: dict[str, ProbeResult]) -> None:
        with self._listeners_lock:
            listeners = [listener for _, listener in self._request_end_listeners]
        for listener in listeners:
            try:
                listener(request_id, dict(results))
            except Exception:  # noqa: BLE001 -- a misbehaving listener must not affect end_request
                logger.exception("on_request_end listener %r raised for request_id=%r", listener, request_id)

    def __enter__(self) -> Router:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown(wait=True)

    def get_probe(self, request_id: str, extraction_point_name: str) -> Probe:
        """Look up the live probe instance for one (request, extraction point).

        For introspection and testing; ordinary usage doesn't need it.

        Raises:
            RouterError: the request or extraction point isn't registered.
        """
        with self._lock:
            bindings = self._requests.get(request_id)
            if bindings is None:
                raise RouterError(f"get_probe(): request_id={request_id!r} is not registered. {_NOT_REGISTERED_HINT}")
            binding = bindings.get(extraction_point_name)
            if binding is None:
                raise RouterError(_no_such_point("get_probe()", extraction_point_name, request_id, bindings))
            return binding.probe

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        """Stop every still-registered request's async bindings and shut down the worker pool.

        Call it once, when the host process is stopping; use ``end_request`` for
        per-request cleanup. Requests still registered are dropped without
        calling their probes' ``on_end``. ``shutdown`` is terminal: afterwards
        ``register_request``, ``route`` and ``end_request`` raise
        [`RouterError`][undercurrent.router.RouterError].

        Args:
            wait: ``True`` (default): close every live async binding's queue and
                drain it (like ``end_request`` does), and block until every
                worker has exited. ``False``: discard whatever is queued and
                return promptly; an activation already in progress still runs to
                completion (a Python thread can't be preempted), and its worker
                exits after it.
            timeout: with ``wait=True``, seconds to wait for each binding to
                drain (each binding gets its own budget). Defaults to the router's
                ``drain_timeout``.
        """
        # The timeout budget is per binding, not cumulative: a shared budget exhausted
        # by one slow binding shouldn't make every other one look like it failed to
        # drain. See AsyncWorker.cancel for the wait=False caveat.
        with self._lock:
            self._closed = True
            all_bindings = list(self._requests.values())
            request_ids = list(self._requests.keys())
            self._requests.clear()
            self._request_contexts.clear()

        drain_timeout = timeout if timeout is not None else self._drain_timeout
        for request_id, bindings in zip(request_ids, all_bindings):
            for name, binding in bindings.items():
                if binding.worker is None:
                    continue
                if wait:
                    binding.worker.stop_and_drain(timeout=drain_timeout)
                else:
                    binding.worker.cancel()
                self._metrics_registry.forget(request_id, name)

        self._executor.shutdown(wait=wait)

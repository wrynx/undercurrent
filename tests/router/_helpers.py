"""Test helpers shared by the router tests (not fixtures -- those live in
conftest.py). Import them absolutely: `from tests.router._helpers import X`."""

import threading
import time
from collections.abc import Callable

from undercurrent.core import ActivationRecord, Probe, ProbeResult
from undercurrent.router import MetricsSink
from undercurrent.spec import ExecutionMode, ExtractionPoint, ProbeKind, TensorType, parse_position


def make_extraction_point(
    name="ep-1",
    layer=5,
    tensor=TensorType.RESIDUAL_STREAM,
    position="generated[*]",
    probe_type="trajectory_score",
    probe_kind=ProbeKind.TRAJECTORY,
    execution_mode=ExecutionMode.INLINE,
    stride=None,
    until=None,
    queue_depth=None,
    intervention=None,
) -> ExtractionPoint:
    layers = (layer,) if isinstance(layer, int) else tuple(layer)
    return ExtractionPoint(
        name=name,
        layers=layers,
        tensor_type=tensor,
        position=parse_position(position),
        stride=stride,
        until=until,
        probe_type=probe_type,
        probe_kind=probe_kind,
        execution_mode=execution_mode,
        queue_depth=queue_depth,
        intervention=intervention,
    )


class SlowProbe(Probe):
    """Test double: on_activation sleeps, to prove async dispatch doesn't
    block route()'s caller."""

    probe_kind = "trajectory"

    def __init__(self, delay: float = 0.3) -> None:
        super().__init__()
        self._delay = delay
        self.received = []

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        time.sleep(self._delay)
        self.received.append(record)
        return None

    def on_end(self, request_ctx):
        return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=list(self.received))


class GatedProbe(Probe):
    """Test double: on_activation blocks on an Event until released.

    Used to deterministically force a queue backlog: route one record,
    wait for `started` to confirm the worker has picked it up (so the
    queue is empty and the worker is busy), then push more records to
    force genuine overflow before releasing the gate.
    """

    probe_kind = "trajectory"

    def __init__(self) -> None:
        super().__init__()
        self.received = []
        self.started = threading.Event()
        self._gate = threading.Event()

    def release(self) -> None:
        self._gate.set()

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        self.started.set()
        self._gate.wait(timeout=5)
        self.received.append(record)
        return None

    def on_end(self, request_ctx):
        return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=list(self.received))


class FlakyProbe(Probe):
    """Test double: on_activation raises for records matching `raise_on`,
    otherwise records the activation normally. Used to verify failure
    isolation -- a raising on_activation must not crash the worker pool,
    must not stop subsequent items (including from other requests sharing
    the pool) from being processed, and must be observable via metrics/the
    log sink rather than silently swallowed."""

    probe_kind = "trajectory"

    def __init__(self, raise_on: Callable[[ActivationRecord], bool]) -> None:
        super().__init__()
        self._raise_on = raise_on
        self.received = []

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        if self._raise_on(record):
            raise ValueError(f"synthetic failure for token_pos={record.token_pos}")
        self.received.append(record)
        return None

    def on_end(self, request_ctx):
        return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=list(self.received))


class ImmediateAbortProbe(Probe):
    """Test double: on_activation returns action=ABORT immediately, no delay."""

    probe_kind = "trajectory"

    def __init__(self) -> None:
        super().__init__()
        self.received = []

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        from undercurrent.core import ProbeAction, ProbeSignal

        self.received.append(record)
        return ProbeSignal(action=ProbeAction.ABORT, metadata={"reason": "immediate_abort_probe"})

    def on_end(self, request_ctx):
        return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=list(self.received))


class FakeLogSink:
    """Test double satisfying `undercurrent.router.SupportsLogSink`: records
    every forwarded signal/result in order, without any I/O."""

    def __init__(self, raise_on_signal: bool = False) -> None:
        self.signals = []
        self.results = []
        self._raise_on_signal = raise_on_signal

    def write_signal(self, request_id, extraction_point_name, signal):
        self.signals.append((request_id, extraction_point_name, signal))
        if self._raise_on_signal:
            raise RuntimeError("synthetic log sink failure")

    def write_result(self, request_id, extraction_point_name, result):
        self.results.append((request_id, extraction_point_name, result))


class FakeMetricsSink(MetricsSink):
    """Test double implementing `undercurrent.router.MetricsSink`: records
    every forwarded event in order, without any I/O. `raise_on` names the
    method (e.g. "record_drop") that should raise once called, to verify
    a misbehaving external sink can't affect dispatch."""

    def __init__(self, raise_on: str = "") -> None:
        self.queue_depths = []
        self.drops = []
        self.activations = []
        self.errors = []
        self._raise_on = raise_on

    def record_queue_depth(self, request_id, extraction_point_name, depth):
        self.queue_depths.append((request_id, extraction_point_name, depth))
        if self._raise_on == "record_queue_depth":
            raise RuntimeError("synthetic metrics sink failure")

    def record_drop(self, request_id, extraction_point_name):
        self.drops.append((request_id, extraction_point_name))
        if self._raise_on == "record_drop":
            raise RuntimeError("synthetic metrics sink failure")

    def record_activation(self, request_id, extraction_point_name, latency_seconds):
        self.activations.append((request_id, extraction_point_name, latency_seconds))
        if self._raise_on == "record_activation":
            raise RuntimeError("synthetic metrics sink failure")

    def record_probe_error(self, request_id, extraction_point_name):
        self.errors.append((request_id, extraction_point_name))
        if self._raise_on == "record_probe_error":
            raise RuntimeError("synthetic metrics sink failure")


def wait_until(predicate: Callable[[], bool], timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll `predicate` until it's truthy or `timeout` elapses. Returns
    whether it ever became true -- used instead of a fixed `sleep` to wait
    for async worker threads to finish processing without hardcoding a
    duration that's either too slow (flaky) or too long (slow suite)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()

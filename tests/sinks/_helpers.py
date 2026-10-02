"""Test helpers shared by the sinks tests (not fixtures -- those live in
conftest.py). Import them absolutely: `from tests.sinks._helpers import X`."""

import threading
import time

from undercurrent.core import ActivationRecord, Probe, ProbeAction, ProbeResult, ProbeSignal, RequestContext
from undercurrent.spec import ExecutionMode, ExtractionPoint, ProbeKind, TensorType, parse_position


def make_extraction_point(
    name="ep-1",
    layer=5,
    tensor=TensorType.RESIDUAL_STREAM,
    position="generated[*]",
    probe_type="emitting",
    probe_kind=ProbeKind.TRAJECTORY,
    execution_mode=ExecutionMode.ASYNC,
    stride=None,
    until=None,
    queue_depth=None,
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
    )


def make_record(point, token_pos=0, request_id="req-1", tensor=None) -> ActivationRecord:
    return ActivationRecord(
        request_id=request_id,
        extraction_point_name=point.name,
        layer=point.layers[0],
        token_pos=token_pos,
        tensor_type=point.tensor_type.value,
        tensor=tensor if tensor is not None else [float(token_pos)],
        is_generated=True,
    )


class EmittingProbe(Probe):
    """Test double: emits a ProbeSignal from every `on_activation` call
    (unlike `TrajectoryScoreProbe`, which only emits once a threshold is
    crossed) -- makes signal-forwarding assertions straightforward."""

    probe_kind = "trajectory"

    def __init__(self) -> None:
        super().__init__()
        self.received: list[ActivationRecord] = []

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        self.received.append(record)
        return ProbeSignal(action=ProbeAction.CONTINUE, metadata={"token_pos": record.token_pos})

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"count": len(self.received)},
        )


class SlowProbe(Probe):
    """Test double: on_activation sleeps before returning a signal, so a
    binding processes records slowly regardless of the log sink attached."""

    probe_kind = "trajectory"

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self._delay = delay
        self.received: list[ActivationRecord] = []

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        time.sleep(self._delay)
        self.received.append(record)
        return ProbeSignal(action=ProbeAction.CONTINUE)

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"count": len(self.received)},
        )


class RecordingSink:
    """Test double implementing the LogSink interface: records every call,
    with an optional injected delay/failure to simulate a slow or
    misbehaving sink without needing a real file or network endpoint."""

    def __init__(self, delay: float = 0.0, raise_on_signal: bool = False) -> None:
        self._delay = delay
        self._raise_on_signal = raise_on_signal
        self.signals: list[tuple[str, str, ProbeSignal]] = []
        self.results: list[tuple[str, str, ProbeResult]] = []
        self._lock = threading.Lock()

    def write_signal(self, request_id: str, extraction_point_name: str, signal: ProbeSignal) -> None:
        if self._delay:
            time.sleep(self._delay)
        if self._raise_on_signal:
            raise RuntimeError("simulated log sink failure")
        with self._lock:
            self.signals.append((request_id, extraction_point_name, signal))

    def write_result(self, request_id: str, extraction_point_name: str, result: ProbeResult) -> None:
        with self._lock:
            self.results.append((request_id, extraction_point_name, result))

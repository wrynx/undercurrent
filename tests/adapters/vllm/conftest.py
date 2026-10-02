import pytest

from undercurrent.core import ActivationRecord, Probe, ProbeResult, RequestContext
from undercurrent.router import ProbeFactory, Router


class RecordingProbe(Probe):
    """Test double: records every ActivationRecord it sees; optionally
    returns an abort signal once a threshold of activations is reached."""

    probe_kind = "trajectory"

    def __init__(self, abort_after: int | None = None) -> None:
        super().__init__()
        self._abort_after = abort_after
        self.received = []

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def on_activation(self, record: ActivationRecord):
        from undercurrent.core import ProbeAction, ProbeSignal

        self.received.append(record)
        if self._abort_after is not None and len(self.received) >= self._abort_after:
            return ProbeSignal(action=ProbeAction.ABORT)
        return None

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=list(self.received))


class SingleShotRecordingProbe(Probe):
    probe_kind = "single_shot"

    def __init__(self) -> None:
        super().__init__()
        self.received = []

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def on_activation(self, record: ActivationRecord):
        self.received.append(record)
        return None

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        verdict = self.received[0] if self.received else None
        return ProbeResult(request_ctx.request_id, self.extraction_point_name, verdict=verdict)


@pytest.fixture
def probe_registry():
    return {
        "recording": ProbeFactory(RecordingProbe),
        "recording_abort_after_1": ProbeFactory(RecordingProbe, {"abort_after": 1}),
        "single_shot_recording": ProbeFactory(SingleShotRecordingProbe),
    }


@pytest.fixture
def router(probe_registry):
    r = Router(probe_registry=probe_registry)
    yield r
    r.shutdown(wait=True)

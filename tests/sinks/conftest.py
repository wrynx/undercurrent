import pytest

from tests.sinks._helpers import EmittingProbe, SlowProbe
from undercurrent.core import RequestContext
from undercurrent.router import ProbeFactory, Router


@pytest.fixture
def probe_registry():
    return {
        "emitting": ProbeFactory(EmittingProbe, {}),
        "slow": ProbeFactory(SlowProbe, {}),
    }


@pytest.fixture
def make_request_ctx():
    def _make(request_id="req-1"):
        return RequestContext(request_id=request_id, prompt_metadata={}, extraction_point_config=None)

    return _make


@pytest.fixture
def router(probe_registry):
    r = Router(probe_registry)
    yield r
    r.shutdown()

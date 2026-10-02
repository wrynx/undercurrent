import pytest

from undercurrent.core import ActivationRecord, RequestContext
from undercurrent.core.examples import MLPClassifierProbe, TrajectoryScoreProbe
from undercurrent.router import ProbeFactory


@pytest.fixture
def probe_registry():
    return {
        "mlp_classifier": ProbeFactory(MLPClassifierProbe, {"num_classes": 2}),
        "trajectory_score": ProbeFactory(TrajectoryScoreProbe, {"threshold": 0.8}),
    }


@pytest.fixture
def make_record():
    def _make(
        request_id="req-1",
        extraction_point_name="ep-1",
        layer=5,
        token_pos=0,
        tensor_type="residual_stream",
        tensor=None,
        is_generated=True,
    ):
        return ActivationRecord(
            request_id=request_id,
            extraction_point_name=extraction_point_name,
            layer=layer,
            token_pos=token_pos,
            tensor_type=tensor_type,
            tensor=tensor if tensor is not None else [1.0, 2.0],
            is_generated=is_generated,
        )

    return _make


@pytest.fixture
def make_request_ctx():
    def _make(request_id="req-1", extraction_point_config=None):
        return RequestContext(
            request_id=request_id, prompt_metadata={}, extraction_point_config=extraction_point_config
        )

    return _make

import pytest

from undercurrent.core import ActivationRecord, RequestContext


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
            tensor=tensor if tensor is not None else [1.0, 2.0, 3.0, 4.0],
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

import pytest

from undercurrent.core import ActivationRecord


@pytest.fixture
def make_record():
    def _make(
        request_id="req-1",
        extraction_point_name="ep-1",
        layer=0,
        token_pos=0,
        tensor_type="residual_stream",
        tensor=None,
        is_generated=False,
    ):
        return ActivationRecord(
            request_id=request_id,
            extraction_point_name=extraction_point_name,
            layer=layer,
            token_pos=token_pos,
            tensor_type=tensor_type,
            tensor=tensor if tensor is not None else [1.0, 2.0, 3.0],
            is_generated=is_generated,
        )

    return _make

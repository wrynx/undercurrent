import time

import pytest

from undercurrent.spec import ActivationRecord


def test_construction_with_list_as_tensor():
    # Framework-agnostic: a plain list stands in for numpy/torch here so
    # this package never has to import either.
    record = ActivationRecord(
        request_id="req-1",
        extraction_point_name="probe_a",
        layer=5,
        token_pos=42,
        tensor_type="residual_stream",
        tensor=[0.1, 0.2, 0.3],
        is_generated=True,
    )
    assert record.tensor == [0.1, 0.2, 0.3]
    assert record.timestamp <= time.time()


def test_default_timestamp_is_set():
    before = time.time()
    record = ActivationRecord(
        request_id="req-1",
        extraction_point_name="probe_a",
        layer=0,
        token_pos=0,
        tensor_type="residual_stream",
        tensor=[1.0],
        is_generated=False,
    )
    after = time.time()
    assert before <= record.timestamp <= after


def test_explicit_timestamp_respected():
    record = ActivationRecord(
        request_id="req-1",
        extraction_point_name="probe_a",
        layer=0,
        token_pos=0,
        tensor_type="residual_stream",
        tensor=[1.0],
        is_generated=False,
        timestamp=123.456,
    )
    assert record.timestamp == 123.456


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", ""),
        ("extraction_point_name", ""),
        ("layer", -1),
        ("token_pos", -1),
        ("tensor", None),
    ],
)
def test_invalid_fields_rejected(field, value):
    kwargs = {
        "request_id": "req-1",
        "extraction_point_name": "probe_a",
        "layer": 0,
        "token_pos": 0,
        "tensor_type": "residual_stream",
        "tensor": [1.0],
        "is_generated": False,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        ActivationRecord(**kwargs)


def test_metadata_excludes_tensor():
    record = ActivationRecord(
        request_id="req-1",
        extraction_point_name="probe_a",
        layer=5,
        token_pos=42,
        tensor_type="residual_stream",
        tensor=object(),  # deliberately not JSON-serializable
        is_generated=True,
        timestamp=1.0,
    )
    meta = record.metadata()
    assert "tensor" not in meta
    assert meta == {
        "request_id": "req-1",
        "extraction_point_name": "probe_a",
        "layer": 5,
        "token_pos": 42,
        "tensor_type": "residual_stream",
        "is_generated": True,
        "timestamp": 1.0,
    }


def test_accepts_numpy_like_and_torch_like_tensor_without_importing_them():
    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

    record = ActivationRecord(
        request_id="req-1",
        extraction_point_name="probe_a",
        layer=1,
        token_pos=1,
        tensor_type="residual_stream",
        tensor=FakeTensor((1, 4096)),
        is_generated=False,
    )
    assert record.tensor.shape == (1, 4096)

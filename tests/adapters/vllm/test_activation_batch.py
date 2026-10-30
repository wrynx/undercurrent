"""The packed worker -> driver activation format (activation_batch.py)."""

import pytest

torch = pytest.importorskip("torch")

from undercurrent.adapters.vllm.activation_batch import (  # noqa: E402
    BATCH_MARKER,
    pack_activation_batch,
    unpack_activation_batch,
    unpack_activation_batches,
)
from undercurrent.spec import ActivationRecord  # noqa: E402


def record(request_id, layer, tensor, token_pos=5, tensor_type="residual_stream", is_generated=True):
    return ActivationRecord(
        request_id=request_id,
        extraction_point_name=f"ep-{layer}",
        layer=layer,
        token_pos=token_pos,
        tensor_type=tensor_type,
        tensor=tensor,
        is_generated=is_generated,
    )


def as_tuples(raws):
    return [
        (r["request_id"], r["extraction_point_name"], r["layer"], r["token_pos"], r["tensor_type"], r["is_generated"])
        for r in raws
    ]


def test_round_trip_preserves_fields_values_and_order_within_a_width():
    records = [
        record("a", 0, torch.arange(4, dtype=torch.float32)),
        record("b", 3, torch.arange(4, 8, dtype=torch.bfloat16), token_pos=9, tensor_type="mlp_out"),
        record("a", 1, torch.full((4,), -1.5), is_generated=False),
    ]
    raws = unpack_activation_batch(pack_activation_batch(records))

    assert as_tuples(raws) == [
        ("a", "ep-0", 0, 5, "residual_stream", True),
        ("b", "ep-3", 3, 9, "mlp_out", True),
        ("a", "ep-1", 1, 5, "residual_stream", False),
    ]
    for raw, original in zip(raws, records):
        assert raw["tensor"].dtype == torch.float32
        assert raw["tensor"].shape == (4,)
        assert torch.equal(raw["tensor"], original.tensor.float())


def test_different_widths_travel_in_separate_groups():
    batch = pack_activation_batch([record("a", 0, torch.ones(3)), record("a", 1, torch.ones(5))])
    assert sorted(group["numel"] for group in batch["groups"]) == [3, 5]
    assert sorted(r["tensor"].numel() for r in unpack_activation_batch(batch)) == [3, 5]


def test_batch_contains_only_rpc_safe_types():
    batch = pack_activation_batch([record("a", 0, torch.ones(2))])
    (group,) = batch["groups"]
    assert isinstance(group["data"], bytes)
    assert all(isinstance(group[k], list) for k in ("request_id", "layer", "token_pos", "is_generated"))


def test_unpacked_tensors_are_writable_and_independent():
    raws = unpack_activation_batch(
        pack_activation_batch([record("a", 0, torch.zeros(2)), record("b", 0, torch.zeros(2))])
    )
    raws[0]["tensor"].add_(1.0)
    assert torch.equal(raws[1]["tensor"], torch.zeros(2))


def test_empty_batch():
    assert unpack_activation_batch(pack_activation_batch([])) == []


def test_unknown_format_version_fails_loudly():
    batch = pack_activation_batch([])
    batch[BATCH_MARKER] = 99
    with pytest.raises(ValueError, match="different undercurrent versions"):
        unpack_activation_batch(batch)


def test_unpack_batches_accepts_one_batch_or_one_per_worker_and_passes_others_through():
    one = pack_activation_batch([record("a", 0, torch.ones(2))])
    two = pack_activation_batch([record("b", 0, torch.ones(2))])
    assert [as_tuples(w) for w in unpack_activation_batches(one)] == [as_tuples(unpack_activation_batch(one))]
    assert [[r["request_id"] for r in w] for w in unpack_activation_batches([one, two])] == [["a"], ["b"]]
    already_records = [{"request_id": "x"}]
    assert unpack_activation_batches(already_records) is already_records

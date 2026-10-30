"""Wire format for activations crossing from the vLLM worker to the driver.

On the cross-process polling path (see worker_extension.py's "CROSS-PROCESS
ACTIVATION POLLING"), `pop_pending_activations()` returns one batch per call
instead of one dict per record. Records whose tensors have the same number of
elements share a group: their tensors are stacked into one float32 buffer and
sent as `bytes`, with the other fields as parallel lists. Everything in a batch
is a str, int, bool, list or bytes, which vLLM's msgpack RPC encoder carries
natively.

Earlier versions sent each tensor as a nested Python list (`.tolist()`), which
built and serialised one Python float per element: about half a millisecond
per record for a 3,584-wide hidden state, paid on the engine's thread. Packing
costs one memcpy per group on each side.

`unpack_activation_batches()` turns a `collective_rpc` result (one batch, or one
per worker) back into per-worker lists of the record dicts `adapter.py` merges,
deduplicates and routes. Any other value is passed through unchanged, so
callers that already hold record dicts keep working.

Internal: not part of the public API.
"""

from __future__ import annotations

from typing import Any

BATCH_MARKER = "undercurrent_activation_batch"
_FORMAT_VERSION = 1
_FIELDS = ("request_id", "extraction_point_name", "layer", "token_pos", "tensor_type", "is_generated")


def pack_activation_batch(records: list[Any]) -> dict[str, Any]:
    """Pack `ActivationRecord`s (CPU tensors) into one RPC-safe batch."""
    import torch

    groups: dict[int, list[Any]] = {}
    for record in records:
        groups.setdefault(record.tensor.numel(), []).append(record)
    packed = []
    for numel, group in groups.items():
        stacked = torch.stack([r.tensor.reshape(numel) for r in group]).to(torch.float32)
        packed.append(
            {
                "numel": numel,
                "data": stacked.contiguous().numpy().tobytes(),
                **{name: [getattr(r, name) for r in group] for name in _FIELDS},
            }
        )
    return {BATCH_MARKER: _FORMAT_VERSION, "groups": packed}


def is_activation_batch(value: Any) -> bool:
    return isinstance(value, dict) and BATCH_MARKER in value


def unpack_activation_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """One batch back into record dicts whose `tensor` is a 1-D float32 CPU tensor."""
    import torch

    version = batch[BATCH_MARKER]
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"activation batch format {version!r} is not supported (expected {_FORMAT_VERSION}); "
            "the driver and the vLLM worker are running different undercurrent versions"
        )
    records = []
    for group in batch["groups"]:
        count = len(group["request_id"])
        # bytearray: one writable copy, so probes get ordinary (writable) tensors.
        rows = torch.frombuffer(bytearray(group["data"]), dtype=torch.float32).reshape(count, group["numel"])
        for i in range(count):
            record = {name: group[name][i] for name in _FIELDS}
            record["tensor"] = rows[i]
            records.append(record)
    return records


def unpack_activation_batches(pending: Any) -> Any:
    """Unpack a `collective_rpc` result: a batch, or a list with one batch per worker.

    Returns a list with one list of record dicts per batch; anything that isn't
    a batch (or a list of them) is returned unchanged.
    """
    if is_activation_batch(pending):
        return [unpack_activation_batch(pending)]
    if isinstance(pending, list) and pending and all(is_activation_batch(item) for item in pending):
        return [unpack_activation_batch(item) for item in pending]
    return pending

"""Unit tests for ProbingWorkerExtension's registration/bookkeeping RPC
methods, WITHOUT a real vLLM Worker or torch.

These exercise everything that doesn't require an actual forward pass to
fire (registration, duplicate/kv rejection, unregistration, abort polling,
router binding, hook-installation wiring reaching a fake model). Hook
FIRING on a real tensor is covered only by the (skipped-without-vllm)
integration test, since `_extract_captured_tensor` needs `torch.Tensor` to
type-check against.

`ProbingWorkerExtension` is normally mixed into vLLM's `Worker` via
multiple inheritance, which is what supplies `self.model_runner` at
runtime. Here we set `.model_runner` directly on a bare instance to a
lightweight fake -- legitimate, since the extension only ever accesses it
through plain attribute access / duck typing, never `super()`.
"""

import pytest

from tests.adapters.vllm._helpers import FakeModelRunner, FakeModule, make_extraction_point
from undercurrent.adapters.vllm.worker_extension import ProbingWorkerExtension, WorkerExtractionError
from undercurrent.spec import InterventionMode, InterventionPolicy, TensorType


def make_extension(num_layers: int = 2) -> ProbingWorkerExtension:
    ext = ProbingWorkerExtension()
    ext.model_runner = FakeModelRunner(num_layers=num_layers)  # type: ignore[attr-defined]
    return ext


def test_register_extraction_installs_hooks_on_every_layer():
    ext = make_extension(num_layers=3)
    eps = [make_extraction_point(name="ep-1", layer=0)]
    ext.register_extraction("req-1", eps, prompt_len=5)

    for layer in ext.model_runner.model.model.layers:
        assert len(layer.hooks) == 1  # residual_stream hook
        assert len(layer.self_attn.hooks) == 1
        assert len(layer.mlp.hooks) == 1
    assert ext._hooks_installed is True
    assert ext._execute_model_wrapped is True


def test_register_extraction_hook_installation_is_idempotent_across_requests():
    ext = make_extension(num_layers=2)
    ext.register_extraction("req-1", [make_extraction_point(name="ep-1", layer=0)], prompt_len=3)
    ext.register_extraction("req-2", [make_extraction_point(name="ep-2", layer=1)], prompt_len=4)

    # Hooks installed exactly once per layer, not once per request.
    for layer in ext.model_runner.model.model.layers:
        assert len(layer.hooks) == 1


def test_register_extraction_rejects_duplicate_request_id():
    ext = make_extension()
    eps = [make_extraction_point(name="ep-1", layer=0)]
    ext.register_extraction("req-1", eps, prompt_len=5)
    with pytest.raises(WorkerExtractionError):
        ext.register_extraction("req-1", eps, prompt_len=5)


def test_register_extraction_rejects_kv_tensor_type():
    ext = make_extension()
    eps = [make_extraction_point(name="ep-1", layer=0, tensor=TensorType.KV, position=0)]
    with pytest.raises(WorkerExtractionError):
        ext.register_extraction("req-1", eps, prompt_len=5)
    # Rejected atomically -- no partial registration left behind.
    assert "req-1" not in ext._requests
    assert not ext._mapper.is_registered("req-1")


def test_unregister_extraction_is_idempotent_and_clears_state():
    ext = make_extension()
    eps = [make_extraction_point(name="ep-1", layer=0)]
    ext.register_extraction("req-1", eps, prompt_len=5)
    ext.unregister_extraction("req-1")
    assert "req-1" not in ext._requests
    assert not ext._mapper.is_registered("req-1")
    ext.unregister_extraction("req-1")  # no error calling twice
    ext.unregister_extraction("never-registered")  # no error


def test_pop_pending_aborts_drains_and_clears():
    ext = make_extension()
    ext._pending_aborts.update({"req-1", "req-2"})
    popped = set(ext.pop_pending_aborts())
    assert popped == {"req-1", "req-2"}
    assert ext.pop_pending_aborts() == []


def test_unregister_extraction_clears_pending_abort_for_that_request():
    ext = make_extension()
    ext._pending_aborts.add("req-1")
    ext.unregister_extraction("req-1")
    assert "req-1" not in ext._pending_aborts


def test_pop_pending_activations_drains_and_clears():
    from undercurrent.spec import ActivationRecord

    ext = make_extension()
    record = ActivationRecord(
        request_id="req-1",
        extraction_point_name="ep-1",
        layer=0,
        token_pos=0,
        tensor_type="residual_stream",
        tensor="row0",
        is_generated=True,
    )
    ext._pending_activations.append(record)
    popped = ext.pop_pending_activations()
    assert popped == [
        {
            "request_id": "req-1",
            "extraction_point_name": "ep-1",
            "layer": 0,
            "token_pos": 0,
            "tensor_type": "residual_stream",
            "tensor": "row0",
            "is_generated": True,
        }
    ]
    assert ext.pop_pending_activations() == []


def test_register_extraction_accepts_plain_dicts_and_reconstructs_extraction_points():
    """extraction_points may arrive as plain dicts, not ExtractionPoint
    instances, when they crossed a real collective_rpc boundary -- see
    register_extraction()'s docstring and adapter.py's generate(), which
    pre-serializes with extraction_point_to_dict() for exactly this
    reason (vLLM's typed-arg RPC decoder can't reconstruct a plain
    @dataclass like ExtractionPoint from its encoded form)."""
    from undercurrent.spec import extraction_point_to_dict

    ext = make_extension()
    point = make_extraction_point(name="ep-1", layer=0, position="prompt[-1]")
    as_dict = extraction_point_to_dict(point)
    assert isinstance(as_dict, dict)

    ext.register_extraction("req-1", [as_dict], prompt_len=5)

    reconstructed = ext._requests["req-1"].extraction_points[0]
    assert reconstructed.name == point.name
    assert reconstructed.tensor_type == point.tensor_type
    assert reconstructed.probe_type == point.probe_type
    assert reconstructed.layers == point.layers


def test_bind_router_stores_reference_and_returns_true():
    ext = make_extension()
    sentinel_router = object()
    result = ext.bind_router(sentinel_router)
    assert result is True
    assert ext._router is sentinel_router


def test_decoder_layer_discovery_falls_back_through_known_attribute_paths():
    ext = ProbingWorkerExtension()

    class GptStyleModel:
        def __init__(self, layers):
            class Transformer:
                pass

            self.transformer = Transformer()
            self.transformer.h = layers

    class Runner:
        pass

    runner = Runner()
    runner.model = GptStyleModel([FakeModule(), FakeModule()])
    runner.execute_model = lambda scheduler_output, *a, **kw: "ok"
    ext.model_runner = runner  # type: ignore[attr-defined]

    ext.register_extraction("req-1", [make_extraction_point(name="ep-1", layer=1)], prompt_len=2)
    assert len(runner.model.transformer.h[0].hooks) == 1
    assert len(runner.model.transformer.h[1].hooks) == 1


def test_decoder_layer_discovery_raises_clearly_when_no_known_shape_matches():
    ext = ProbingWorkerExtension()

    class UnknownModel:
        pass

    class Runner:
        pass

    runner = Runner()
    runner.model = UnknownModel()
    ext.model_runner = runner  # type: ignore[attr-defined]

    with pytest.raises(WorkerExtractionError):
        ext.register_extraction("req-1", [make_extraction_point(name="ep-1", layer=0)], prompt_len=2)


def test_register_extraction_warns_once_for_block_until_signal(caplog):
    # See worker_extension.py's "INTERVENTION TIMEOUT LIMITATIONS" section:
    # block_until_signal isn't safe under continuous batching (the bounded
    # wait stalls the whole scheduler step, not just the triggering
    # request), so this adapter logs a loud one-time warning the first time
    # it sees an extraction point configured that way.
    ext = make_extension(num_layers=1)
    point = make_extraction_point(
        name="ep-1", intervention=InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=100)
    )

    with caplog.at_level("WARNING", logger="undercurrent.adapters.vllm.worker_extension"):
        ext.register_extraction("req-1", [point], prompt_len=2)

    warnings = [r for r in caplog.records if "block_until_signal" in r.message]
    assert len(warnings) == 1
    assert "ep-1" in warnings[0].message

    # Second request with the same problematic policy: no additional warning.
    point2 = make_extraction_point(
        name="ep-2", intervention=InterventionPolicy(mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=100)
    )
    with caplog.at_level("WARNING", logger="undercurrent.adapters.vllm.worker_extension"):
        ext.register_extraction("req-2", [point2], prompt_len=2)

    warnings = [r for r in caplog.records if "block_until_signal" in r.message]
    assert len(warnings) == 1


def test_register_extraction_does_not_warn_for_reject_or_unset_intervention():
    ext = make_extension(num_layers=1)
    point_a = make_extraction_point(name="ep-1", intervention=InterventionPolicy(mode=InterventionMode.REJECT))
    point_b = make_extraction_point(name="ep-2")  # intervention=None

    ext.register_extraction("req-1", [point_a, point_b], prompt_len=2)

    assert ext._warned_block_until_signal is False

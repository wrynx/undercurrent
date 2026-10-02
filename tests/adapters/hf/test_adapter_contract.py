import threading
import time

import pytest

from tests.adapters.hf._helpers import SlowThenContinueProbe
from undercurrent.adapters.hf import HFAdapterLimitationError
from undercurrent.router import ProbeFactory, Router
from undercurrent.spec import ExecutionMode, ExtractionPoint, ProbeKind, TensorType, parse_position


def _make_point(name="ep-1", tensor=TensorType.RESIDUAL_STREAM, layer=0):
    return ExtractionPoint(
        name=name,
        layers=(layer,),
        tensor_type=tensor,
        position=parse_position("generated[*]"),
        stride=None,
        until=None,
        probe_type="slow_then_continue",
        probe_kind=ProbeKind.TRAJECTORY,
        execution_mode=ExecutionMode.INLINE,
        queue_depth=None,
    )


def test_kv_tensor_type_rejected_at_registration(adapter):
    point = _make_point(tensor=TensorType.KV)
    with pytest.raises(HFAdapterLimitationError, match="tensor_type"):
        adapter.register_extraction("req-1", [point])


def test_generate_before_register_extraction_raises(adapter, prompt):
    router = Router({})
    with pytest.raises(HFAdapterLimitationError, match="no prior register_extraction"):
        adapter.generate("req-unregistered", prompt, {"max_new_tokens": 1}, router)
    router.shutdown()


def test_double_register_extraction_for_same_pending_request_raises(adapter):
    point = _make_point()
    adapter.register_extraction("req-1", [point])
    with pytest.raises(HFAdapterLimitationError, match="already pending"):
        adapter.register_extraction("req-1", [point])
    adapter.unregister_extraction("req-1")


def test_concurrent_generate_calls_rejected(adapter, prompt):
    router = Router({"slow_then_continue": ProbeFactory(SlowThenContinueProbe, {"delay": 0.5})})
    point = _make_point()
    adapter.register_extraction("req-1", [point])

    errors = []

    def _run():
        try:
            adapter.generate("req-1", prompt, {"max_new_tokens": 3, "do_sample": False}, router)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.05)  # let the first generate() actually start (past register_request)

    adapter.register_extraction("req-2", [_make_point(name="ep-2")])
    with pytest.raises(HFAdapterLimitationError, match="another request is still active"):
        adapter.generate("req-2", prompt, {"max_new_tokens": 1}, router)

    t.join(timeout=5)
    assert not errors
    router.shutdown()


def test_unregister_extraction_is_idempotent(adapter):
    adapter.unregister_extraction("does-not-exist")  # must not raise


def test_forward_wrapper_keeps_the_model_signature(adapter, prompt):
    # transformers' generate() validates model_kwargs against
    # inspect.signature(model.forward); 4.40 rejects `attention_mask` if the
    # step-counting wrapper hides the real signature behind *args/**kwargs.
    import inspect

    router = Router({"slow_then_continue": ProbeFactory(SlowThenContinueProbe, {"delay": 0.0})})
    adapter.register_extraction("req-sig", [_make_point()])
    try:
        adapter.generate("req-sig", prompt, {"max_new_tokens": 1, "do_sample": False}, router)
    finally:
        router.shutdown()
    params = inspect.signature(adapter._model.forward).parameters
    assert "attention_mask" in params
    assert "input_ids" in params

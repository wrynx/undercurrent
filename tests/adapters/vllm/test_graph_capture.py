"""Graph-mode residual capture (graph_capture.py), without vLLM.

The runner patch is exercised against a fake GPUModelRunner whose model
implements vLLM's EAGLE-3 aux-output interface; the worker extension's graph
mode against a fake runner whose execute_model calls the aux callback the way
the patched `_model_forward` does. The real compiled/captured path is covered
on GPU (benchmarks/run.py --graph-capture, fidelity checks).
"""

import os
import types

import pytest

from tests.adapters.vllm._helpers import FakeModelRunner, make_extraction_point
from undercurrent.adapters.vllm.graph_capture import (
    AUX_CALLBACK_ATTR,
    GRAPH_CAPTURE_ENV,
    LAYERS_ATTR,
    GraphCaptureError,
    configure_runner,
    decode_layers,
    encode_layers,
    install_runner_patch,
    unsupported_extraction_points,
)
from undercurrent.spec import TensorType


class AuxModel:
    def __init__(self):
        self.aux_layers = None

    def set_aux_hidden_state_layers(self, layers):
        self.aux_layers = layers


class Runner:
    def __init__(self, model=None, use_aux=False):
        self.model = model if model is not None else AuxModel()
        self.use_aux_hidden_state_outputs = use_aux
        self.forward_calls = 0

    def get_model(self):
        return self.model

    def _model_forward(self, **kwargs):
        self.forward_calls += 1
        return ("hidden", ["aux-a", "aux-b"])

    def load_model(self):
        self.loaded = True


def test_layers_round_trip_sorted_and_deduplicated():
    assert encode_layers([7, 3, 7, 0]) == "0,3,7"
    assert decode_layers("0,3,7") == (0, 3, 7)
    assert decode_layers("") == ()


def test_only_configured_residual_stream_points_are_capturable():
    ok = make_extraction_point(name="ok", layer=[3, 7])
    other_type = make_extraction_point(name="mlp", layer=3, tensor=TensorType.MLP_OUT)
    other_layer = make_extraction_point(name="far", layer=[3, 9])
    problems = unsupported_extraction_points([ok, other_type, other_layer], (3, 7))
    assert len(problems) == 2
    assert "'mlp'" in problems[0] and "residual_stream" in problems[0]
    assert "'far'" in problems[1] and "[9]" in problems[1]
    assert unsupported_extraction_points([ok], (3, 7)) == []


def test_configure_runner_turns_on_aux_outputs_and_forwards_them_to_the_callback():
    runner = Runner()
    configure_runner(runner, [7, 3])

    assert runner.model.aux_layers == (4, 8)  # aux index k + 1 is the output of decoder layer k
    assert runner.use_aux_hidden_state_outputs is True
    assert getattr(runner, LAYERS_ATTR) == (3, 7)

    assert runner._model_forward(x=1) == ("hidden", ["aux-a", "aux-b"])  # output unchanged, no callback yet
    received = []
    setattr(runner, AUX_CALLBACK_ATTR, received.append)
    assert runner._model_forward(x=1) == ("hidden", ["aux-a", "aux-b"])
    assert received == [["aux-a", "aux-b"]]


def test_configure_runner_with_no_layers_leaves_the_model_alone():
    runner = Runner()
    configure_runner(runner, [])
    received = []
    setattr(runner, AUX_CALLBACK_ATTR, received.append)
    runner._model_forward()
    assert runner.model.aux_layers is None
    assert runner.use_aux_hidden_state_outputs is False
    assert getattr(runner, LAYERS_ATTR) == ()
    assert received == []


def test_configure_runner_refuses_models_without_aux_outputs_and_speculative_decoding():
    with pytest.raises(GraphCaptureError, match="EAGLE-3"):
        configure_runner(Runner(model=object()), [1])
    with pytest.raises(GraphCaptureError, match="speculative decoding"):
        configure_runner(Runner(use_aux=True), [1])


def test_install_runner_patch_configures_after_load_and_is_idempotent():
    class PatchedRunner(Runner):
        pass

    install_runner_patch(PatchedRunner, (2,))
    install_runner_patch(PatchedRunner, (2,))  # a second plugin call must not wrap twice
    runner = PatchedRunner()
    runner.load_model()
    assert runner.loaded is True
    assert runner.model.aux_layers == (3,)
    runner._model_forward()
    assert runner.forward_calls == 1


# --- worker extension in graph mode ------------------------------------------------


@pytest.fixture
def graph_env(monkeypatch):
    monkeypatch.setenv(GRAPH_CAPTURE_ENV, "1,3")


class GraphStepRunner(FakeModelRunner):
    """execute_model calls the aux callback, as the patched _model_forward does after a forward."""

    def __init__(self, layers=(1, 3)):
        super().__init__(num_layers=4)
        setattr(self, LAYERS_ATTR, tuple(layers))
        self.aux = None

        def _execute_model(scheduler_output, *a, **kw):
            callback = getattr(self, AUX_CALLBACK_ATTR, None)
            if callback is not None and self.aux is not None:
                callback(self.aux)
            return "ok"

        self.execute_model = _execute_model


def test_worker_graph_mode_installs_no_hooks_and_captures_from_aux_outputs(graph_env, router):
    torch = pytest.importorskip("torch")
    from undercurrent.adapters.vllm.worker_extension import ProbingWorkerExtension
    from undercurrent.core import RequestContext

    ext = ProbingWorkerExtension()
    runner = GraphStepRunner()
    ext.model_runner = runner  # type: ignore[attr-defined]
    ext.bind_router(router)
    ep = make_extraction_point(name="ep", layer=[1, 3], probe_type="recording")
    for rid in ("req-a", "req-b"):
        ext.register_extraction(rid, [ep], prompt_len=2)
        router.register_request(rid, [ep], RequestContext(rid, {}, None))

    assert all(not layer.hooks for layer in runner.model.model.layers)  # no forward hooks in graph mode
    assert getattr(runner, AUX_CALLBACK_ATTR) is not None

    # Two decode rows, padded to 4 (CUDA-graph padding rows never match).
    runner.input_batch = types.SimpleNamespace(req_ids=["req-a", "req-b"], num_computed_tokens_cpu=[2, 2])
    runner.aux = [torch.arange(4.0).repeat(3, 1).T * 1.0, torch.arange(4.0).repeat(3, 1).T * 10.0]
    ext.model_runner.execute_model(types.SimpleNamespace(num_scheduled_tokens={"req-a": 1, "req-b": 1}))

    a = router.get_probe("req-a", "ep").received
    b = router.get_probe("req-b", "ep").received
    assert [(r.layer, r.tensor[0].item()) for r in a] == [(1, 0.0), (3, 0.0)]
    assert [(r.layer, r.tensor[0].item()) for r in b] == [(1, 1.0), (3, 10.0)]
    router.end_request("req-a")
    router.end_request("req-b")


def test_worker_graph_mode_rejects_other_tensor_types_and_layers(graph_env):
    from undercurrent.adapters.vllm.worker_extension import ProbingWorkerExtension, WorkerExtractionError

    ext = ProbingWorkerExtension()
    ext.model_runner = GraphStepRunner()  # type: ignore[attr-defined]
    with pytest.raises(WorkerExtractionError, match="only residual_stream"):
        ext.register_extraction("r1", [make_extraction_point(name="m", layer=1, tensor=TensorType.MLP_OUT)], 2)
    with pytest.raises(WorkerExtractionError, match=r"layers \[2\]"):
        ext.register_extraction("r2", [make_extraction_point(name="x", layer=2)], 2)
    assert ext._requests == {}


def test_worker_refuses_graph_mode_when_the_plugin_did_not_configure_the_runner(graph_env):
    from undercurrent.adapters.vllm.worker_extension import ProbingWorkerExtension, WorkerExtractionError

    ext = ProbingWorkerExtension()
    ext.model_runner = FakeModelRunner(num_layers=4)  # type: ignore[attr-defined]  # no LAYERS_ATTR
    with pytest.raises(WorkerExtractionError, match="general_plugins"):
        ext.register_extraction("r1", [make_extraction_point(name="ep", layer=1)], 2)


# --- adapter ------------------------------------------------------------------------


def test_adapter_graph_capture_sets_and_clears_the_environment(monkeypatch):
    from undercurrent.adapters.vllm.adapter import VLLMAdapterLimitationError, VLLMEngineAdapter

    for name in (GRAPH_CAPTURE_ENV, "VLLM_DISABLE_COMPILE_CACHE"):
        monkeypatch.setenv(name, "")  # registers the original value, so it's restored afterwards
        monkeypatch.delenv(name)
    adapter = VLLMEngineAdapter()
    adapter._configure_graph_capture([5, 1])
    assert os.environ[GRAPH_CAPTURE_ENV] == "1,5"
    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"

    adapter.register_extraction("ok", [make_extraction_point(name="ep", layer=5)])
    with pytest.raises(VLLMAdapterLimitationError, match="not capturable in graph mode"):
        adapter.register_extraction("bad", [make_extraction_point(name="ep", layer=5, tensor=TensorType.ATTN_OUT)])

    adapter._configure_graph_capture(None)
    assert GRAPH_CAPTURE_ENV not in os.environ
    adapter.register_extraction("any", [make_extraction_point(name="ep", layer=2, tensor=TensorType.ATTN_OUT)])

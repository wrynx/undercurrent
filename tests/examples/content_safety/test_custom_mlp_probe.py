import pytest
import torch
from content_safety_demo import CustomMLPProbe, custom_mlp
from content_safety_demo.custom_mlp import UnsafeCheckpointError, _classifier_cache, _load_cached_classifier
from content_safety_demo.models import SafetyClassifierHead

LAYER = 5
INPUT_DIM = 4
HIDDEN_SIZE = 8

# The demo refuses to load checkpoints on torch older than 2.6 (CVE-2025-32434),
# so every test that loads one needs a new enough torch. The refusal itself is
# tested below with a patched version, on any torch.
needs_safe_torch = pytest.mark.skipif(
    custom_mlp._torch_version_tuple() < custom_mlp._MIN_SAFE_TORCH,
    reason="the content-safety demo needs torch>=2.6 to load checkpoints",
)


def _write_checkpoint(tmp_path, input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE, seed=0, wrapped=False):
    torch.manual_seed(seed)
    model = SafetyClassifierHead(input_dim, hidden_size)
    path = tmp_path / "checkpoint.pt"
    state_dict = model.state_dict()
    torch.save({"state_dict": state_dict} if wrapped else state_dict, path)
    return str(path), model


@needs_safe_torch
def test_loads_checkpoint_and_produces_matching_scores(tmp_path, make_record, make_request_ctx):
    path, model = _write_checkpoint(tmp_path)
    tensor = [3.0, -1.0, 2.0, 0.5]
    with torch.no_grad():
        expected_score = float(model(torch.tensor(tensor, dtype=torch.float32)))

    probe = CustomMLPProbe.spawn("req-1", "ep-1", layer=LAYER, model_path=path, threshold=0.5)
    probe.on_start(make_request_ctx())
    signal = probe.on_activation(make_record(layer=LAYER, tensor=tensor))
    assert signal.confidence == expected_score
    result = probe.on_end(make_request_ctx())
    assert result.verdict["score"] == expected_score
    assert result.verdict["flagged"] == (expected_score > 0.5)


@needs_safe_torch
def test_tolerates_state_dict_wrapper(tmp_path, make_record, make_request_ctx):
    path, _model = _write_checkpoint(tmp_path, wrapped=True)
    probe = CustomMLPProbe.spawn("req-1", "ep-1", layer=LAYER, model_path=path)
    probe.on_start(make_request_ctx())
    signal = probe.on_activation(make_record(layer=LAYER, tensor=[1.0, 2.0, 3.0, 4.0]))
    assert 0.0 <= signal.confidence <= 1.0


@needs_safe_torch
def test_checkpoint_is_cached_across_spawns(tmp_path, make_record, make_request_ctx):
    path, _ = _write_checkpoint(tmp_path)
    _classifier_cache.clear()

    probe_a = CustomMLPProbe.spawn("req-a", "ep-1", layer=LAYER, model_path=path)
    probe_b = CustomMLPProbe.spawn("req-b", "ep-1", layer=LAYER, model_path=path)
    assert probe_a._classifier is probe_b._classifier
    assert len(_classifier_cache) == 1


@needs_safe_torch
def test_wrong_layer_raises(tmp_path, make_record, make_request_ctx):
    path, _ = _write_checkpoint(tmp_path)
    probe = CustomMLPProbe.spawn("req-1", "ep-1", layer=LAYER, model_path=path)
    probe.on_start(make_request_ctx())
    try:
        probe.on_activation(make_record(layer=LAYER + 1, tensor=[1.0, 2.0, 3.0, 4.0]))
        assert False, "expected ValueError"
    except ValueError:
        pass


@needs_safe_torch
def test_wrong_activation_dim_raises(tmp_path, make_record, make_request_ctx):
    path, _ = _write_checkpoint(tmp_path, input_dim=INPUT_DIM)
    probe = CustomMLPProbe.spawn("req-1", "ep-1", layer=LAYER, model_path=path)
    probe.on_start(make_request_ctx())
    try:
        probe.on_activation(make_record(layer=LAYER, tensor=[1.0, 2.0]))  # wrong dim
        assert False, "expected ValueError"
    except ValueError:
        pass


@needs_safe_torch
def test_on_end_fallback_when_on_activation_never_called(make_request_ctx, tmp_path):
    path, _ = _write_checkpoint(tmp_path)
    probe = CustomMLPProbe.spawn("req-1", "ep-1", layer=LAYER, model_path=path)
    probe.on_start(make_request_ctx())
    result = probe.on_end(make_request_ctx())
    assert result.verdict == {"score": None, "flagged": False, "layer": LAYER, "token_pos": None}


@needs_safe_torch
def test_second_on_activation_call_raises(tmp_path, make_record, make_request_ctx):
    path, _ = _write_checkpoint(tmp_path)
    probe = CustomMLPProbe.spawn("req-1", "ep-1", layer=LAYER, model_path=path)
    probe.on_start(make_request_ctx())
    probe.on_activation(make_record(layer=LAYER, tensor=[1.0, 2.0, 3.0, 4.0]))
    try:
        probe.on_activation(make_record(layer=LAYER, tensor=[1.0, 2.0, 3.0, 4.0]))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


@needs_safe_torch
def test_load_cached_classifier_infers_shape_from_state_dict(tmp_path):
    path, _ = _write_checkpoint(tmp_path, input_dim=6, hidden_size=10)
    _classifier_cache.clear()
    model = _load_cached_classifier(path, "cpu")
    assert model.net[0].in_features == 6
    assert model.net[0].out_features == 10


# --- Safe (weights_only) checkpoint loading --------------------------------


def _pwn(marker_path):
    with open(marker_path, "w") as f:
        f.write("pwned")
    return {}


class _MaliciousPayload:
    """Unpickling this calls `_pwn(marker_path)` -- stands in for an arbitrary
    `os.system` call in a crafted checkpoint downloaded from the internet."""

    def __init__(self, marker_path):
        self.marker_path = marker_path

    def __reduce__(self):
        return (_pwn, (self.marker_path,))


class _CustomObject:
    def __init__(self):
        self.weight = torch.zeros(1)


@needs_safe_torch
def test_malicious_checkpoint_is_rejected_without_side_effect(tmp_path):
    marker = tmp_path / "pwned.txt"
    path = tmp_path / "malicious.pt"
    torch.save({"state_dict": _MaliciousPayload(str(marker))}, path)
    try:
        with pytest.raises(UnsafeCheckpointError, match=r"torch\.save\(model\.state_dict\(\), path\)"):
            _load_cached_classifier(str(path), "cpu")
    finally:
        _classifier_cache.clear()
        # Checked even if the load raised something else: the payload must never run.
        assert not marker.exists(), "malicious checkpoint's __reduce__ payload was executed"


@needs_safe_torch
def test_checkpoint_with_custom_object_is_rejected(tmp_path):
    path = tmp_path / "custom.pt"
    torch.save(_CustomObject(), path)
    try:
        with pytest.raises(UnsafeCheckpointError, match="Only plain tensor state_dicts"):
            _load_cached_classifier(str(path), "cpu")
    finally:
        _classifier_cache.clear()


@needs_safe_torch
def test_checkpoint_that_is_not_a_tensor_dict_is_rejected(tmp_path):
    path = tmp_path / "list.pt"
    torch.save([torch.zeros(1)], path)
    try:
        with pytest.raises(UnsafeCheckpointError, match="not a tensor state_dict"):
            _load_cached_classifier(str(path), "cpu")
    finally:
        _classifier_cache.clear()


@needs_safe_torch
@pytest.mark.parametrize("wrapped", [False, True])
def test_plain_and_wrapped_state_dicts_load_weights_only(tmp_path, wrapped):
    path, model = _write_checkpoint(tmp_path, wrapped=wrapped)
    try:
        loaded = _load_cached_classifier(path, "cpu")
        for key, value in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value)
    finally:
        _classifier_cache.clear()


def test_old_torch_refuses_to_load(tmp_path, monkeypatch):
    path, _ = _write_checkpoint(tmp_path)
    monkeypatch.setattr(custom_mlp, "_torch_version_tuple", lambda: (2, 5))
    try:
        with pytest.raises(RuntimeError, match="CVE-2025-32434"):
            _load_cached_classifier(path, "cpu")
    finally:
        _classifier_cache.clear()

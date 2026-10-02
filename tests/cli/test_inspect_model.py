"""`undercurrent inspect-model` on tiny configs saved to disk (offline, no weights)."""

import json

import pytest

from undercurrent.cli import inspect_model as inspect_mod
from undercurrent.cli import main

TENSOR_TYPES = ["residual_stream", "attn_out", "mlp_out", "kv", "final_norm"]


def run_json(capsys, *argv):
    assert main(["inspect-model", *map(str, argv), "--format", "json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_gpt2_text_output(gpt2_dir, capsys):
    assert main(["inspect-model", str(gpt2_dir)]) == 0
    out = capsys.readouterr().out
    assert "class:       GPT2LMHeadModel (model_type=gpt2)" in out
    assert "layers:      2" in out
    assert "hidden size: 32" in out
    assert "heads:       2" in out
    assert "vocab:       100" in out
    assert "meta device" in out
    assert "tensor_type      hf      vllm" in out
    assert 'Probe it with layers 0..1, e.g. layers: [1], position: "prompt[-1]"' in out


def test_gpt2_json_shape(gpt2_dir, capsys):
    report = run_json(capsys, gpt2_dir)
    assert report["model"] == str(gpt2_dir)
    assert report["model_class"] == "GPT2LMHeadModel"
    assert report["model_type"] == "gpt2"
    assert (report["num_layers"], report["hidden_size"], report["num_attention_heads"], report["vocab_size"]) == (
        2,
        32,
        2,
        100,
    )
    assert report["verified"] is True
    assert report["weights_loaded"] is False
    assert report["note"] is None
    assert report["suggested"] == {"layers": [1], "tensor_type": "residual_stream", "position": "prompt[-1]"}
    assert list(report["support"]) == ["hf", "vllm"]
    hf, vllm = report["support"]["hf"], report["support"]["vllm"]
    assert list(hf) == TENSOR_TYPES
    assert {t: c["status"] for t, c in hf.items()} == {
        "residual_stream": "yes",
        "attn_out": "yes",
        "mlp_out": "yes",
        "kv": "no",
        "final_norm": "no",
    }
    # vLLM hooks `self_attn` only; GPT-2 blocks call it `attn`.
    assert {t: c["status"] for t, c in vllm.items()} == {
        "residual_stream": "yes",
        "attn_out": "no",
        "mlp_out": "yes",
        "kv": "no",
        "final_norm": "yes",
    }
    assert all(c["reason"] for c in [*hf.values(), *vllm.values()] if c["status"] == "no")


def test_llama(llama_dir, capsys):
    report = run_json(capsys, llama_dir, "--backend", "vllm")
    assert report["model_class"] == "LlamaForCausalLM"
    assert report["num_layers"] == 4
    assert report["suggested"]["layers"] == [2]
    assert list(report["support"]) == ["vllm"]
    assert report["support"]["vllm"]["attn_out"]["status"] == "yes"
    assert report["support"]["vllm"]["final_norm"]["status"] == "yes"


def test_backend_hf_only(llama_dir, capsys):
    assert main(["inspect-model", str(llama_dir), "--backend", "hf"]) == 0
    out = capsys.readouterr().out
    assert "tensor_type      hf\n" in out
    # The tmp path itself may contain "vllm", so drop it before checking.
    assert "vllm" not in out.replace(str(llama_dir), "")


def test_parameters_stay_on_meta(gpt2_dir, capsys, monkeypatch):
    built = []
    original = inspect_mod._instantiate_on_meta

    def spy(*args, **kwargs):
        model, note = original(*args, **kwargs)
        built.append(model)
        return model, note

    monkeypatch.setattr(inspect_mod, "_instantiate_on_meta", spy)
    run_json(capsys, gpt2_dir)
    (model,) = built
    params = list(model.parameters())
    assert params
    assert {p.device.type for p in params} == {"meta"}


def test_unsupported_architecture_gives_helpful_error(opt_dir, capsys):
    assert main(["inspect-model", str(opt_dir)]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: could not locate a decoder-layer list on OPTForCausalLM")
    assert "find_decoder_layers()" in captured.err
    assert "Traceback" not in captured.err


def test_falls_back_to_config_when_meta_build_fails(gpt2_dir, capsys, monkeypatch):
    monkeypatch.setattr(inspect_mod, "_instantiate_on_meta", lambda config, **kw: (None, "could not build it"))
    report = run_json(capsys, gpt2_dir)
    assert report["verified"] is False
    assert report["note"] == "could not build it"
    assert report["num_layers"] == 2
    assert report["model_class"] is None  # config.architectures, unset in a bare saved config
    statuses = {t: c["status"] for t, c in report["support"]["hf"].items()}
    assert statuses == {
        "residual_stream": "unverified",
        "attn_out": "unverified",
        "mlp_out": "unverified",
        "kv": "no",
        "final_norm": "no",
    }
    assert main(["inspect-model", str(gpt2_dir)]) == 0
    assert "note:        could not build it" in capsys.readouterr().out


def test_missing_model_directory_is_a_user_error(tmp_path, capsys):
    assert main(["inspect-model", str(tmp_path / "does-not-exist")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_directory_without_known_model_type(tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "definitely_not_a_model"}))
    assert main(["inspect-model", str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("error: ")


@pytest.mark.parametrize("fixture", ["gpt2_dir", "llama_dir"])
def test_emit_spec_round_trips_through_validate(fixture, request, tmp_path, capsys):
    model_dir = request.getfixturevalue(fixture)
    spec_path = tmp_path / "starter.yaml"
    assert main(["inspect-model", str(model_dir), "--emit-spec", str(spec_path)]) == 0
    capsys.readouterr()

    text = spec_path.read_text(encoding="utf-8")
    assert text.startswith("# yaml-language-server: $schema=https://")
    assert main(["validate", "--strict", str(spec_path)]) == 0
    assert capsys.readouterr().out == f"{spec_path}: OK (1 extraction point)\n"

    from undercurrent.spec import load_yaml_file

    (point,) = load_yaml_file(spec_path).extraction_points
    expected_layer = {"gpt2_dir": 1, "llama_dir": 2}[fixture]
    assert point.layers == (expected_layer,)
    assert point.tensor_type.value == "residual_stream"
    assert point.position.to_raw() == "prompt[-1]"
    assert point.probe_type == "my_probe"
    assert point.probe_kind.value == "single_shot"

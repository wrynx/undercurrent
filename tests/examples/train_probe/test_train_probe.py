"""examples/train_probe: collect -> train -> save -> load -> run inline, on CPU and offline.

The model is a tiny random GPT-2 (2 layers, n_embd=32) saved to tmp_path with
a word-level tokenizer over the toy dataset's vocabulary, so every toy prompt
tokenises to distinct ids without a download.
"""

import json
import math
import re
import runpy
from pathlib import Path

import pytest
import torch
from collector import ActivationCollectorProbe
from data_sources import SOURCES, load_examples
from linear_probe import (
    FORMAT_VERSION,
    PROBE_TYPE,
    ProbeHead,
    TrainedLinearProbe,
    load_head,
    load_probe,
    read_meta,
    save_probe,
    spec_for,
)
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from train_probe import auroc, binary_metrics, main, stratified_split
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from undercurrent.core import ActivationRecord, ProbeAction, RequestContext
from undercurrent.model import ProbedModel

USE_PROBE = Path(__file__).resolve().parents[3] / "examples" / "train_probe" / "use_probe.py"
SPLIT_KEYS = {"n", "accuracy", "precision", "recall", "f1", "auroc"}


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    examples, _ = load_examples("toy")
    words = sorted({w for text, _ in examples for w in re.findall(r"\w+|[^\w\s]", text.lower())})
    words += ["<unk>", "<eos>", *sorted(set(re.findall(r"\w+|[^\w\s]", USE_PROBE.read_text(encoding="utf-8").lower())))]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    eos = vocab["<eos>"]
    config = GPT2Config(
        n_layer=2, n_embd=32, n_head=2, n_inner=64, n_positions=64, vocab_size=len(vocab),
        bos_token_id=eos, eos_token_id=eos,
    )  # fmt: skip
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-gpt2")
    GPT2LMHeadModel(config).eval().save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def trained(tiny_model_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("probe")
    metrics = main(
        ["--model", tiny_model_dir, "--dataset", "toy", "--layer", "1", "--epochs", "2",
         "--device", "cpu", "--seed", "0", "--out", str(out)]
    )  # fmt: skip
    return out, metrics


def test_toy_run_writes_metrics_and_probe_files(trained):
    out, returned = trained
    assert {p.name for p in out.iterdir()} >= {"probe.safetensors", "probe.json", "metrics.json"}
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics == json.loads(json.dumps(returned))
    for key in ("model", "dataset", "layer", "position", "probe", "threshold", "num_examples", "label_names"):
        assert key in metrics
    assert metrics["num_examples"] == len(load_examples("toy")[0])
    for split in ("train", "val", "test"):
        assert set(metrics[split]) == SPLIT_KEYS
        assert metrics[split]["n"] > 0
        assert 0.0 <= metrics[split]["accuracy"] <= 1.0
    assert sum(metrics[s]["n"] for s in ("train", "val", "test")) == metrics["num_examples"]


def test_probe_json_metadata(trained):
    out, metrics = trained
    meta = read_meta(out)
    assert meta["format_version"] == FORMAT_VERSION
    assert meta["architecture"] == "linear"
    assert meta["input_dim"] == 32
    assert meta["layers"] == [1]
    assert meta["tensor_type"] == "residual_stream"
    assert meta["position"] == "prompt[-1]"
    assert meta["base_model"] == metrics["model"]
    assert meta["label_names"] == ["non_toxic", "toxic"]
    assert meta["threshold"] == 0.5
    assert meta["metrics"]["test"] == metrics["test"]


def test_load_round_trips_weights(trained, tmp_path):
    out, _ = trained
    head, meta = load_head(out)
    again = tmp_path / "again"
    save_probe(again, head, {k: v for k, v in meta.items() if k not in ("format", "format_version")})
    head2, meta2 = load_head(again)
    for (name, a), (name2, b) in zip(head.state_dict().items(), head2.state_dict().items()):
        assert name == name2
        assert torch.equal(a, b)
    assert meta2 == meta

    factory = load_probe(out)
    assert factory.probe_cls is TrainedLinearProbe
    assert factory.probe_kwargs["threshold"] == 0.5
    assert factory.probe_kwargs["label_names"] == ("non_toxic", "toxic")


def test_loaded_probe_runs_inline_through_probed_model(trained, tiny_model_dir):
    out, _ = trained
    meta = read_meta(out)
    prompt = "Could you tell me more about the weekend trip?"
    with ProbedModel.from_pretrained(tiny_model_dir, spec=spec_for(meta), probes={PROBE_TYPE: load_probe(out)}) as m:
        result = m.generate(prompt, max_new_tokens=3, temperature=0.0)
    verdict = result.probe_results["trained_probe"].verdict
    assert 0.0 <= verdict["score"] <= 1.0
    assert verdict["layer"] == 1
    assert result.aborted == verdict["flagged"]


@pytest.mark.parametrize("threshold, aborts", [(0.0, True), (1.01, False)])
def test_threshold_override_controls_abort(trained, tiny_model_dir, threshold, aborts):
    out, _ = trained
    spec = spec_for(read_meta(out))
    spec["extraction_points"][0]["probe_args"] = {"threshold": threshold}
    with ProbedModel.from_pretrained(tiny_model_dir, spec=spec, probes={PROBE_TYPE: load_probe(out)}) as m:
        result = m.generate("Shut up, you are a worthless idiot.", max_new_tokens=5, temperature=0.0)
    assert result.aborted is aborts
    if aborts:
        assert result.abort_point == "trained_probe"
        # HF checks the stop flag after sampling, so an abort at the prompt's
        # last token still lets the first generated token through.
        assert len(_tokens(tiny_model_dir, result.text)) <= 1


def _tokens(model_dir, text):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)(text)["input_ids"]


def test_use_probe_script_runs(trained, tiny_model_dir, monkeypatch, capsys):
    out, _ = trained
    monkeypatch.setattr("sys.argv", ["use_probe.py", str(out), tiny_model_dir])
    runpy.run_path(str(USE_PROBE), run_name="__main__")
    printed = capsys.readouterr().out
    assert printed.count("score=") == 2
    assert len(USE_PROBE.read_text(encoding="utf-8").splitlines()) <= 25


def test_tampered_input_dim_is_rejected(trained, tmp_path):
    out, _ = trained
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    (tampered / "probe.safetensors").write_bytes((out / "probe.safetensors").read_bytes())
    meta = json.loads((out / "probe.json").read_text())
    meta["input_dim"] = 4096
    (tampered / "probe.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=r"input_dim=4096.*32 input features.*don't belong together"):
        load_probe(tampered)


def test_wrong_format_version_and_missing_files_are_rejected(trained, tmp_path):
    out, _ = trained
    with pytest.raises(FileNotFoundError, match=r"train_probe\.py"):
        load_probe(tmp_path)
    meta = json.loads((out / "probe.json").read_text())
    meta["format_version"] = 99
    (tmp_path / "probe.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format_version=99"):
        load_probe(tmp_path)
    meta["format_version"] = FORMAT_VERSION
    (tmp_path / "probe.json").write_text(json.dumps(meta))
    with pytest.raises(FileNotFoundError, match=r"probe\.safetensors"):
        load_probe(tmp_path)


def test_mlp_probe_saves_and_loads(tmp_path):
    head = ProbeHead(8, "mlp", hidden_dim=5)
    meta = {"layers": [0], "tensor_type": "residual_stream", "position": "prompt[-1]", "base_model": "x",
            "label_names": ["a", "b"], "threshold": 0.7}  # fmt: skip
    save_probe(tmp_path, head, meta)
    loaded, saved_meta = load_head(tmp_path)
    assert saved_meta["architecture"] == "mlp"
    assert saved_meta["hidden_dim"] == 5
    x = torch.randn(3, 8)
    assert torch.allclose(loaded(x), head.eval()(x))


def test_trained_probe_rejects_wrong_feature_count():
    probe = load_probe_from_head(ProbeHead(4))
    with pytest.raises(ValueError, match="3 features but the probe was trained on 4"):
        probe.on_activation(_record(torch.zeros(3)))


def load_probe_from_head(head, threshold=0.5):
    probe = TrainedLinearProbe.spawn("req-1", "ep-1", head=head, threshold=threshold)
    probe.on_start(RequestContext("req-1", {}, None))
    return probe


def _record(tensor, token_pos=0, layer=1):
    return ActivationRecord("req-1", "ep-1", layer, token_pos, "residual_stream", tensor=tensor, is_generated=False)


# --- ActivationCollectorProbe -------------------------------------------------------


def test_collector_stores_detached_cpu_copies():
    probe = ActivationCollectorProbe.spawn("req-1", "ep-1")
    ctx = RequestContext("req-1", {}, None)
    probe.on_start(ctx)
    source = torch.arange(4.0, requires_grad=True) * 1.0
    assert probe.on_activation(_record(source, token_pos=3)) is None
    assert probe.on_activation(_record([5.0, 6.0, 7.0, 8.0], token_pos=4)) is None
    result = probe.on_end(ctx)

    first, second = result.verdict
    assert first["token_pos"] == 3 and first["layer"] == 1 and first["tensor_type"] == "residual_stream"
    assert first["is_generated"] is False
    assert not first["tensor"].requires_grad
    assert first["tensor"].device.type == "cpu"
    assert torch.equal(first["tensor"], torch.arange(4.0))
    assert first["tensor"].data_ptr() != source.data_ptr()  # a copy, not a view
    assert torch.equal(second["tensor"], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    assert result.metadata == {"captured": 2, "dropped": 0, "max_tokens": None}


def test_collector_max_tokens_cap():
    probe = ActivationCollectorProbe.spawn("req-1", "ep-1", max_tokens=2)
    ctx = RequestContext("req-1", {}, None)
    probe.on_start(ctx)
    for pos in range(5):
        probe.on_activation(_record(torch.full((2,), float(pos)), token_pos=pos))
    result = probe.on_end(ctx)
    assert [r["token_pos"] for r in result.verdict] == [0, 1]
    assert result.metadata == {"captured": 2, "dropped": 3, "max_tokens": 2}


def test_collector_rejects_bad_cap():
    with pytest.raises(ValueError, match="max_tokens"):
        ActivationCollectorProbe.spawn("req-1", "ep-1", max_tokens=0)


def test_collector_with_no_activations_returns_empty_verdict():
    probe = ActivationCollectorProbe.spawn("req-1", "ep-1")
    ctx = RequestContext("req-1", {}, None)
    probe.on_start(ctx)
    assert probe.on_end(ctx).verdict == []


# --- metrics, splits, data --------------------------------------------------------------


def _pairwise_auroc(labels, scores):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def test_auroc_matches_pairwise_definition():
    gen = torch.Generator().manual_seed(1)
    labels = torch.randint(0, 2, (60,), generator=gen).tolist()
    scores = (torch.randint(0, 10, (60,), generator=gen) / 10).tolist()  # plenty of ties
    assert auroc(labels, scores) == pytest.approx(_pairwise_auroc(labels, scores))
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert math.isnan(auroc([1, 1], [0.2, 0.3]))


def test_binary_metrics():
    m = binary_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], threshold=0.5)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["auroc"] == 0.75
    assert binary_metrics([1, 1], [0.9, 0.8], 0.5)["auroc"] is None


def test_stratified_split_covers_every_index_once_and_both_classes():
    labels = [0] * 20 + [1] * 10
    train, val, test = stratified_split(labels, seed=0)
    assert sorted(train + val + test) == list(range(30))
    for split in (train, val, test):
        assert {labels[i] for i in split} == {0, 1}


def test_toy_dataset_is_balanced_and_limit_keeps_balance():
    examples, label_names = load_examples("toy")
    assert label_names == ("non_toxic", "toxic")
    assert len(examples) >= 24
    assert sum(y for _, y in examples) * 2 == len(examples)
    limited, _ = load_examples("toy", limit=7)
    assert len(limited) == 7
    assert sum(y for _, y in limited) in (3, 4)


def test_unknown_dataset_names_the_choices():
    with pytest.raises(ValueError, match=r"civil_comments.*toy"):
        load_examples("imdb")
    assert set(SOURCES) == {"toy", "civil_comments"}


def test_no_pickle_loading_in_example():
    example_dir = USE_PROBE.parent
    for path in example_dir.glob("*.py"):
        assert "torch.load" not in path.read_text(encoding="utf-8"), path


def test_abort_signal_at_threshold():
    head = ProbeHead(2)
    with torch.no_grad():
        head.net.weight.zero_()
        head.net.bias.fill_(0.0)  # score is exactly 0.5
    signal = load_probe_from_head(head, threshold=0.5).on_activation(_record(torch.zeros(2)))
    assert signal.action == ProbeAction.ABORT
    assert signal.confidence == 0.5
    signal = load_probe_from_head(head, threshold=0.51).on_activation(_record(torch.zeros(2)))
    assert signal.action == ProbeAction.CONTINUE

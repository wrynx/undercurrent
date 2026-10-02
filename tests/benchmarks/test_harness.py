"""benchmarks/run.py: dry runs write env.json, results.jsonl and summary.csv
exactly as benchmarks/README.md documents them, and `--backend vllm` fails
fast with one clear line when vLLM or a GPU is missing.

The offline tests run the dry run against a tiny random GPT-2 saved to
tmp_path (word-level tokenizer over the harness's prompt text), so they need
no download. The test of the literal acceptance command, which downloads
`hf-internal-testing/tiny-random-gpt2`, is marked `network`.
"""

import csv
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
RUN_PY = ROOT / "benchmarks" / "run.py"
README = ROOT / "benchmarks" / "README.md"
RUN_DIR_RE = re.compile(r"^\d{8}T\d{6}Z-(hf|vllm)-[A-Za-z0-9._-]+$")


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("undercurrent_benchmarks_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def tiny_model_dir(harness, tmp_path_factory):
    words = re.findall(r"\w+|[^\w\s]", harness._PROMPT_TEXT)
    vocab = {w: i for i, w in enumerate(dict.fromkeys(["<unk>", "<eos>", *words]))}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    eos = vocab["<eos>"]
    config = GPT2Config(
        n_layer=3, n_embd=16, n_head=2, n_inner=32, n_positions=64, vocab_size=len(vocab),
        bos_token_id=eos, eos_token_id=eos,
    )  # fmt: skip
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-gpt2")
    GPT2LMHeadModel(config).eval().save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


# ---------------------------------------------------------------------------
# The documented schema (parsed from benchmarks/README.md)
# ---------------------------------------------------------------------------


def _section(heading):
    text = README.read_text(encoding="utf-8")
    assert f"\n### {heading}\n" in text, f"README has no '### {heading}' section"
    body = text.split(f"\n### {heading}\n", 1)[1]
    return re.split(r"\n#{2,3} |\nChanging a field", body, maxsplit=1)[0]


def _table_rows(heading):
    return re.findall(r"^\| `([a-z0-9_]+)` \|(.*)\|$", _section(heading), re.MULTILINE)


def documented_fields(heading):
    return [name for name, _ in _table_rows(heading)]


def documented_env_args():
    row = dict(_table_rows("env.json"))["args"]
    return set(re.findall(r"`([a-z_]+)`", row.split(":", 1)[1]))


def test_readme_documents_every_output_field(harness):
    assert documented_fields("summary.csv") == list(harness.SUMMARY_COLUMNS)
    for heading in ("env.json", "results.jsonl", "summary.csv"):
        fields = documented_fields(heading)
        assert fields and len(fields) == len(set(fields)), heading


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_run_dir(run_dir, harness, *, repeats, num_prompts, max_new_tokens):
    """Check one run directory against the README schema; return (env, rows, summary)."""
    assert RUN_DIR_RE.match(run_dir.name), run_dir.name
    assert sorted(p.name for p in run_dir.iterdir()) == ["env.json", "results.jsonl", "summary.csv"]

    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert set(env) == set(documented_fields("env.json"))
    assert set(env["args"]) == documented_env_args()
    assert env["schema_version"] == harness.SCHEMA_VERSION == 1
    assert env["run_id"] == run_dir.name
    assert env["failed_scenarios"] == []
    assert set(env["gpu"]) == {"count", "devices", "driver_version", "cuda_version"}
    assert set(env["cpu"]) == {"model", "logical_cores"}
    assert set(env["versions"]) == {"undercurrent", "torch", "transformers", "vllm", "numpy"}
    assert env["versions"]["torch"] and env["versions"]["transformers"]

    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    documented = set(documented_fields("results.jsonl"))
    assert rows
    for row in rows:
        assert set(row) == documented, set(row) ^ documented
        assert row["schema_version"] == 1
        assert row["run_id"] == run_dir.name
        assert row["scenario"] in env["scenarios"]
        assert len(row["batch_latencies_s"]) == row["num_batches"]
        assert all(_is_number(x) and x > 0 for x in row["batch_latencies_s"])
        for key in ("latency_p50_s", "latency_p90_s", "latency_p99_s", "latency_mean_s", "wall_time_s"):
            assert _is_number(row[key]) and row[key] > 0, key
        assert row["latency_p50_s"] <= row["latency_p90_s"] <= row["latency_p99_s"]
        # min_new_tokens == max_new_tokens: every request generates exactly that many.
        assert row["output_tokens_counted"] is True
        assert row["output_tokens"] == num_prompts * max_new_tokens
        assert _is_number(row["output_tokens_per_s"])
        # HF measures TTFT/TTLT for every request.
        assert len(row["ttft_s"]) == len(row["ttlt_s"]) == num_prompts
        assert all(0 < a <= b for a, b in zip(row["ttft_s"], row["ttlt_s"]))
        assert row["peak_gpu_mem_bytes"] is None  # CPU
        assert _is_number(row["overhead_pct_latency_p50"])
        assert _is_number(row["overhead_pct_tokens_per_s"])
        assert _is_number(row["overhead_pct_ex_inline_probe_compute"])
        assert row["num_probes"] == row["num_inline_probes"] + row["num_async_probes"]
        if row["scenario"] == "baseline":
            assert row["num_probes"] == 0 and row["probe_activations"] == 0
            for key in ("queue_drops", "probe_errors", "queue_depth_max", "queue_depth_series", "worker_pool_size"):
                assert row[key] is None, key
        else:
            assert row["probe_errors"] == 0
            assert isinstance(row["worker_pool_size"], int)
            assert isinstance(row["queue_drops"], int) and isinstance(row["queue_depth_max"], int)
            assert len(row["queue_depth_series"]) <= harness.QUEUE_DEPTH_SERIES_POINTS
            assert all(len(p) == 2 and p[1] >= 0 for p in row["queue_depth_series"])
            assert (row["probe_activations"] > 0) == (row["num_probes"] > 0)
        if row["scenario"].startswith("async-slow:"):
            assert row["overflow_policy"] == row["scenario"].split(":", 1)[1]
            assert isinstance(row["latency_flat"], bool)
            assert row["queue_depth"] is not None and row["probe_delay_ms"] is not None
        else:
            assert row["overflow_policy"] is None and row["latency_flat"] is None
    assert len({row["prompts_sha256"] for row in rows}) == 1

    per_scenario = {}
    for row in rows:
        per_scenario.setdefault(row["scenario"], []).append(row["repeat"])
    assert list(per_scenario) == env["scenarios"]
    assert all(r == list(range(repeats)) for r in per_scenario.values())

    with (run_dir / "summary.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == documented_fields("summary.csv")
        summary = list(reader)
    assert [s["scenario"] for s in summary] == env["scenarios"]
    for s in summary:
        assert int(s["repeats"]) == repeats
        assert int(s["schema_version"]) == 1
        for key in ("latency_p50_s", "latency_p99_s", "output_tokens_per_s_mean", "ttft_p50_s", "ttlt_p50_s"):
            assert float(s[key]) > 0, key
        assert s["latency_flat"] in ({"true", "false"} if s["scenario"].startswith("async-slow:") else {""})
        assert s["peak_gpu_mem_bytes"] == ""
    return env, rows, summary


# ---------------------------------------------------------------------------
# Dry runs
# ---------------------------------------------------------------------------


def _single_run_dir(out):
    dirs = [p for p in Path(out).iterdir() if p.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


@pytest.mark.slow
def test_dry_run_covers_every_scenario_and_matches_schema(harness, tiny_model_dir, tmp_path):
    assert harness.main(["--backend", "hf", "--dry-run", "--model", tiny_model_dir, "--out", str(tmp_path)]) == 0
    sizes = harness._DRY_RUN_SIZES
    env, rows, _ = validate_run_dir(
        _single_run_dir(tmp_path),
        harness,
        repeats=sizes["repeats"],
        num_prompts=sizes["num_prompts"],
        max_new_tokens=sizes["max_new_tokens"],
    )
    policies = harness.overflow_policy_names()
    assert policies == ["drop_oldest", "drop_newest", "block"]
    assert env["scenarios"] == [
        "baseline", "probed-0", "inline-1", "async-1", "inline-8", "async-8", "inline-8-mlp", "async-8-mlp",
        "mixed", *(f"async-slow:{p}" for p in policies),
    ]  # fmt: skip
    assert env["dry_run"] is True and env["device"] == "cpu"
    by_name = {}
    for row in rows:
        by_name.setdefault(row["scenario"], []).append(row)
    assert by_name["mixed"][0]["probe_impls"] == "mlp+norm"
    assert by_name["mixed"][0]["num_inline_probes"] == by_name["mixed"][0]["num_async_probes"] == 4
    # Inline probes run on the generation path; async ones don't.
    assert all(r["inline_probe_compute_s"] > 0 for r in by_name["inline-8"])
    assert all(r["inline_probe_compute_s"] == 0 for r in by_name["async-8"])
    # Every activation routed to the slow probe is either processed or
    # dropped by the overflow policy; `block` never drops.
    routed = by_name["async-1"][0]["probe_activations"]
    for policy in policies:
        for row in by_name[f"async-slow:{policy}"]:
            assert row["probe_activations"] + row["queue_drops"] == routed
            assert row["queue_depth_max"] <= row["queue_depth"]
    assert all(r["queue_drops"] == 0 for r in by_name["async-slow:block"])


@pytest.mark.slow
def test_isolated_scenarios_run_in_subprocesses(harness, tiny_model_dir, tmp_path):
    argv = ["--dry-run", "--model", tiny_model_dir, "--out", str(tmp_path), "--isolate", "process"]
    assert harness.main([*argv, "--scenarios", "baseline,inline-1,async-slow:drop_newest"]) == 0
    env, _, _ = validate_run_dir(_single_run_dir(tmp_path), harness, repeats=2, num_prompts=4, max_new_tokens=8)
    assert env["args"]["isolate"] == "process"


@pytest.mark.slow
@pytest.mark.network
def test_acceptance_command_dry_run_downloads_tiny_model(harness, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(RUN_PY), "--backend", "hf", "--dry-run", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    run_dir = _single_run_dir(tmp_path)
    assert proc.stdout.strip() == str(run_dir)
    assert run_dir.name.endswith("-hf-hf-internal-testing--tiny-random-gpt2")
    validate_run_dir(run_dir, harness, repeats=2, num_prompts=4, max_new_tokens=8)


# ---------------------------------------------------------------------------
# Fail fast
# ---------------------------------------------------------------------------


def test_vllm_backend_without_vllm_fails_fast(harness, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(harness, "_module_available", lambda name: name != "vllm")
    assert harness.main(["--backend", "vllm", "--model", "some/model", "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "needs vLLM" in err and "undercurrent[vllm]" in err
    assert "Traceback" not in err
    assert list(tmp_path.iterdir()) == []


def test_vllm_backend_without_gpu_fails_fast(harness, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(harness, "_module_available", lambda name: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert harness.main(["--backend", "vllm", "--model", "some/model", "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "needs a CUDA GPU" in err and "Traceback" not in err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(torch.cuda.is_available(), reason="on a GPU host this would start a real vLLM benchmark")
def test_vllm_cli_on_this_host_exits_with_one_line(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(RUN_PY), "--backend", "vllm", "--model", "gpt2", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert proc.stderr.strip().startswith("benchmarks/run.py: error: --backend vllm needs")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--backend", "vllm", "--dry-run"], "--dry-run is CPU-only"),
        (["--backend", "hf"], "--model is required"),
        (["--backend", "hf", "--model", "m", "--batch-size", "4"], "--batch-size must be 1"),
        (["--backend", "hf", "--model", "m", "--scenarios", "inline-x"], "unknown scenario 'inline-x'"),
        (["--backend", "hf", "--model", "m", "--scenarios", "async-slow:sometimes"], "unknown overflow policy"),
        (["--backend", "hf", "--model", "m", "--scenarios", "inline-0"], "use probed-0"),
        (["--backend", "hf", "--model", "m", "--scenarios", "baseline,baseline"], "listed twice"),
        (["--backend", "hf", "--model", "m", "--vllm-arg", "dtype=half"], "--vllm-arg only applies"),
    ],
)
def test_usage_errors_are_one_line(harness, argv, message, tmp_path, capsys):
    assert harness.main([*argv, "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_async_slow_expands_to_the_routers_overflow_policies(harness):
    from undercurrent.router.overflow import OverflowPolicy

    units = harness.parse_scenarios("async-slow", slow_queue_depth=3)
    assert [u.name for u in units] == [f"async-slow:{p.value}" for p in OverflowPolicy]
    assert all(u.queue_depth == 3 and u.slots == (harness.ProbeSlot("sleep", "async"),) for u in units)


def test_build_spec_is_valid_and_spreads_layers(harness):
    from undercurrent.spec import parse_dict

    (scenario,) = harness.parse_scenarios("async-8-mlp", slow_queue_depth=2)
    spec = parse_dict(harness.build_spec(scenario, num_layers=4))
    points = list(spec)
    assert len(points) == 8
    assert sorted({layer for p in points for layer in p.layers}) == [0, 1, 2, 3]
    assert all(p.probe_type == "bench_mlp" and p.execution_mode.value == "async" for p in points)


def test_percentile_matches_linear_interpolation(harness):
    assert harness.percentile([], 50) is None
    assert harness.percentile([3.0], 99) == 3.0
    assert harness.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert harness.percentile([4.0, 1.0, 3.0, 2.0], 90) == pytest.approx(3.7)


def test_downsample_series_keeps_peaks_and_bounds_length(harness):
    series = [(i / 1000, i % 7) for i in range(5000)]
    out = harness.downsample_series(series, points=50)
    assert len(out) <= 50
    assert max(d for _, d in out) == 6
    assert harness.downsample_series([(0.0, 1)]) == [[0.0, 1]]


def test_model_slug(harness):
    assert harness.model_slug("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama--Llama-3.1-8B-Instruct"
    assert harness.model_slug("/tmp/my model/") == "tmp--my-model"

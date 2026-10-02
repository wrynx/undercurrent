# Benchmarks: probing overhead

`benchmarks/run.py` measures what Undercurrent's probing costs compared with
plain generation: 0, 1 and N probes, inline and async, on the Hugging Face
transformers and vLLM backends. It also has a backpressure scenario that runs
a deliberately slow async probe under each router overflow policy. Results are
machine-readable; the [output schema](#output-schema) below is a contract that
the docs' benchmark page reads.

No numbers are committed here. Results come from running the harness on real
hardware.

## Quick start: CPU dry run

```bash
pip install -e .            # from the repo root; or `pip install undercurrent`
python benchmarks/run.py --backend hf --dry-run --out /tmp/bench
```

`--dry-run` forces CPU, the `hf-internal-testing/tiny-random-gpt2` model (a
small download from the Hugging Face Hub) and tiny sizes (4 prompts of 16 tokens, 8 new
tokens, 2 warmup batches, 2 repeats). Unless you pass `--scenarios`, it runs
every scenario, `async-slow` included. It checks that the harness works and
writes all three output files. **Its timings mean nothing**: the model is
random and tiny, and CPU timings at this size are mostly noise. You may pass
`--model` with a local tiny checkpoint to run the dry run offline.

The harness needs only the standard library plus what `pip install
undercurrent` already installs (torch, transformers). `--backend vllm` also
needs vLLM and a CUDA GPU. Without either it exits at once with a one-line
error (exit code 2).

## Running on GPUs

Run from the repo root at the commit you want to measure, with undercurrent
installed from that checkout (`pip install -e .`). `env.json` records the git
SHA and whether the tree had uncommitted changes. Llama 3.1 is a gated model:
accept its license on the Hugging Face Hub and log in first (`hf auth login`,
`huggingface-cli login` on older `huggingface_hub`, or set `HF_TOKEN`).

### Recommended matrix

**1. vLLM, Llama-3.1-8B-Instruct, 1×A100-80GB or 1×H100.** Run it inside
your existing vLLM environment (vLLM is never installed by undercurrent; the
supported range is in `docs/compatibility.md`). Run it twice, once for
throughput (batch 32) and once for latency (batch 1):

```bash
SCENARIOS=baseline,probed-0,inline-1,async-1,inline-8,async-8,inline-8-mlp,async-8-mlp,mixed,async-slow

python benchmarks/run.py --backend vllm --model meta-llama/Llama-3.1-8B-Instruct \
  --scenarios $SCENARIOS --num-prompts 256 --batch-size 32 \
  --input-len 512 --max-new-tokens 256 --warmup 4 --repeats 5 --seed 0 \
  --worker-pool-size 256 \
  --vllm-arg gpu_memory_utilization=0.85 --vllm-arg max_model_len=4096 \
  --out benchmarks/results

python benchmarks/run.py --backend vllm --model meta-llama/Llama-3.1-8B-Instruct \
  --scenarios $SCENARIOS --num-prompts 64 --batch-size 1 \
  --input-len 512 --max-new-tokens 256 --warmup 4 --repeats 5 --seed 0 \
  --vllm-arg gpu_memory_utilization=0.85 --vllm-arg max_model_len=4096 \
  --out benchmarks/results
```

`--worker-pool-size 256` = batch size 32 × 8 async probes. Each live async
binding pins one router pool thread (see `docs/production/async-execution.md`,
"Bindings pin threads"). With the default pool (at most 32 threads), 256
bindings would mostly wait for a thread, and the benchmark would measure that
instead of the probes.

Each vLLM scenario runs in its own subprocess (`--isolate auto`), because an
engine adapter binds one Router for its lifetime and freeing a vLLM engine's
GPU memory in-process is unreliable. Every scenario therefore loads the model once,
which adds load time but doesn't affect the measurements.

**2. HF transformers, gpt2, one small GPU (T4, L4 or A10G).**

```bash
python benchmarks/run.py --backend hf --model gpt2 --device cuda \
  --scenarios $SCENARIOS --num-prompts 64 \
  --input-len 128 --max-new-tokens 128 --warmup 4 --repeats 5 --seed 0 \
  --out benchmarks/results
```

**Optional 3. HF transformers, Llama-3.1-8B-Instruct in bf16 on an A100/H100:**
the same command with `--model meta-llama/Llama-3.1-8B-Instruct --dtype
bfloat16 --input-len 512 --max-new-tokens 256`.

The HF backend generates one prompt at a time (`ProbedModel`'s HF backend has
`max_concurrency=1`), so `--batch-size` must be 1 there. The baseline also
runs one prompt at a time, so the comparison is like for like.

A run prints its output directory on stdout and exits 0. If a scenario fails,
the harness logs the traceback, records the failure in `env.json`
(`failed_scenarios`), keeps measuring the other scenarios, writes the files
and exits 1.

### Handing results over

Each run writes one directory, `<out>/<run_id>/`. Copy it, unchanged, into the
reports directory the benchmark page is built from:

```bash
mkdir -p "$REPORTS_DIR/benchmarks"
cp -r benchmarks/results/<run_id> "$REPORTS_DIR/benchmarks/"
```

`REPORTS_DIR` is wherever you keep raw benchmark runs outside the repository.
Copy whole directories only, never edited files, and include failed
or surprising runs too: the page must show what was measured. Don't commit
`benchmarks/results/`; it is in `.gitignore`.

## Command-line options

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend {hf,vllm}` | `hf` | Inference backend |
| `--model` | required (dry run: `hf-internal-testing/tiny-random-gpt2`) | HF hub id or local path |
| `--num-prompts` | 64 | Prompts per repeat |
| `--batch-size` | hf: 1 (the only allowed value); vllm: 16 | Prompts per `generate()` call |
| `--max-new-tokens` | 128 | Tokens generated per prompt, exactly (see methodology) |
| `--input-len` | 128 | Prompt length in tokens |
| `--scenarios` | `baseline,inline-1,async-1,inline-8,async-8,mixed` | Comma-separated, see [Scenarios](#scenarios) |
| `--warmup` | 4 | Untimed batches per scenario, before the timed repeats |
| `--repeats` | 3 | Timed passes over all prompts, per scenario |
| `--out` | `benchmarks/results` | Output root directory |
| `--dry-run` | off | CPU smoke run, see above |
| `--seed` | 0 | Seeds prompt selection, torch, the MLP probe weights and vLLM |
| `--probe-delay-ms` | 50 | `async-slow`: sleep per activation |
| `--slow-queue-depth` | 4 | `async-slow`: the probe's `queue_depth` |
| `--flat-tolerance-pct` | 10 | `async-slow`: threshold for `latency_flat` |
| `--worker-pool-size` | router default | `Router(worker_pool_size=...)` for probed scenarios |
| `--device` | `cuda` if available, else `cpu` | hf only: torch device |
| `--dtype {auto,float32,float16,bfloat16}` | `auto` | Model dtype (hf: `from_pretrained`; vllm: engine `dtype`) |
| `--vllm-arg KEY=VALUE` | none | vllm only, repeatable: engine arg passed to both the baseline `vllm.LLM` and the probed engine. VALUE is parsed as JSON when it can be |
| `--isolate {auto,process,none}` | `auto` | Run each scenario in its own subprocess (auto: yes for vllm, no for hf) |

## Scenarios

| Scenario | Probes | What it isolates |
| --- | --- | --- |
| `baseline` | none, and no Undercurrent code at all: plain `model.generate` (hf) or plain `vllm.LLM.generate` (vllm) | The reference every overhead is computed against |
| `probed-0` | a `ProbedModel` with an empty spec | The adapter's always-on cost (hooks, request bookkeeping) with nothing to route. On vLLM it also absorbs the difference between `vllm.LLM` and the async engine the adapter uses |
| `inline-N` | N inline norm probes | Inline routing on the generation path |
| `async-N` | N async norm probes | Async enqueueing on the generation path, plus the end-of-request drain |
| `inline-N-mlp`, `async-N-mlp` | N MLP probes | Same machinery as the norm variants with a costlier probe, which separates probe compute from machinery |
| `mixed` | 4 inline + 4 async; each group alternates norm and MLP | A guard plus background observers |
| `async-slow` | 1 async sleep probe, `queue_depth=--slow-queue-depth`, run once per overflow policy: `async-slow:drop_oldest`, `async-slow:drop_newest`, `async-slow:block` | Backpressure (opt-in) |
| `async-slow:<policy>` | as above, one policy only | |

`N` can be any positive integer. The extraction points of a scenario are
spread evenly over the model's layers (point `i` of `N` is at layer
`floor((i + 0.5) × num_layers / N) mod num_layers`). Every point captures
`residual_stream` at `position: "generated[*]"` (every generated token) with
`probe_kind: trajectory`, so inline and async scenarios run identical probe
code.

### The probes

All three never intervene (`on_activation` returns `None`) and time their own
`on_activation`:

- **norm**: the L2 norm of the activation, converted to a Python float. The
  cheapest realistic probe: one reduction and a host sync.
- **mlp**: a random-weight MLP head `d → 256 → 1` (ReLU, sigmoid), weights
  seeded from `--seed` and built once per run on the activation's device and
  dtype. Roughly what a trained classifier probe costs.
- **sleep**: `time.sleep(--probe-delay-ms)` per activation. Used only by
  `async-slow`.

### The backpressure scenario (`async-slow`)

A probe that is slower than generation is run async with a small queue, once
per router `OverflowPolicy`. The claim under test: with `drop_oldest` and
`drop_newest`, a slow observe-mode probe never stalls token generation, and it
loses activations instead (`queue_drops`). With `block`, generation is
expected to slow down to the probe's pace; that is that policy's documented
contract.

Read three things:

- **`ttlt_p50_s`** (time to last token, hf only) against `baseline`: whether
  token generation stayed flat. `latency_flat` compares this (or
  `latency_p50_s` when TTLT isn't measurable) with the baseline, using
  `--flat-tolerance-pct`.
- **`latency_p50_s`**: end to end. It also includes `Router.end_request`
  draining what is still queued (at most `queue_depth` items per binding,
  bounded by the router's `drain_timeout`), so expect it to exceed TTLT by up
  to about `queue_depth × probe_delay`.
- **`queue_depth_series`, `queue_depth_max`, `queue_drops`**: how full the
  queues ran and how much the policy dropped.

Make the probe clearly slower than one decode step (`--probe-delay-ms` above
the per-token latency of the baseline), or the queue never fills.

## Methodology

- **Identical prompts.** Prompts are cut from a fixed passage at offsets drawn
  from `--seed`, tokenised to `--input-len` tokens. Every scenario gets the
  same list. `prompts_sha256` in each row proves it, and the harness refuses
  to write results if scenarios saw different prompts.
- **Identical output lengths.** Greedy decoding (temperature 0) with
  `min_new_tokens = max_new_tokens` (vLLM: `min_tokens = max_tokens`,
  `ignore_eos=True`), so every request generates exactly `--max-new-tokens`
  tokens and tokens/s compares like with like. None of the probes abort.
- **Warmup.** Each scenario runs `--warmup` untimed batches after loading and
  before its timed repeats. The first scenario in a process also pays
  one-time costs (CUDA context, kernel autotuning, allocator growth), so keep
  `--warmup` at 2 or more. On HF the scenarios share one process and run in
  the order given, so put `baseline` first and look at `probed-0` as a
  sanity check: it should sit close to `baseline`.
- **Seeds.** `--seed` fixes the prompts, the MLP weights, vLLM's engine seed,
  and `torch.manual_seed(seed + repeat)` before each repeat.
- **Repeats.** Each repeat is one full pass over all prompts and produces one
  row in `results.jsonl`. `summary.csv` pools all repeats' batch latencies
  before taking percentiles. Use at least 5 repeats for published numbers and
  check `output_tokens_per_s_std`.
- **Latency.** `latency_*` is the wall-clock time of one `generate()` call on
  one batch of `--batch-size` prompts (on HF, one prompt). On HF, CUDA is
  synchronised before the clock stops.
- **TTFT and TTLT** are measured on HF only, with a token-timing streamer
  attached in every HF scenario, `baseline` included, so its small cost is the
  same everywhere. `ProbedModel`'s vLLM path returns whole outputs, so on
  vLLM these fields are null.
- **Peak GPU memory** is `torch.cuda.max_memory_allocated` per repeat, on HF
  with CUDA. It is null on CPU and on vLLM (the engine preallocates its KV
  cache from `gpu_memory_utilization`, partly in another process, so the
  harness process's allocator stats would mislead).
- **Separate probe compute from machinery overhead.** The overhead of a
  probed scenario has two parts: the probe's own work, and the cost of the
  machinery (hooks, routing, queues, workers, the end-of-request drain).
  Don't report a probed-scenario overhead as "the cost of Undercurrent"
  without saying which probes ran. To tell the parts apart:
  - compare `inline-N` with `inline-N-mlp` (same machinery, different probe
    cost), and `probed-0` with `baseline` (machinery with no probes);
  - `inline_probe_compute_s` is the time spent inside inline probes'
    `on_activation`, which sits on the generation path.
    `overhead_pct_ex_inline_probe_compute` subtracts it from the wall time.
    This is an estimate: on CUDA, the probe's host sync also waits for
    kernels already queued, so it can attribute some model time to the probe.
    Async probe compute (`probe_compute_s` − `inline_probe_compute_s`) runs
    on router threads, off the generation path, but competes for the same
    CPU and GPU.
- **vLLM baseline caveat.** `baseline` is plain `vllm.LLM` (the synchronous
  engine). `ProbedModel` runs vLLM's async engine with the adapter's worker
  extension, and the adapter forces `VLLM_USE_V2_MODEL_RUNNER=0`. Report
  probe overhead relative to `probed-0` as well as to `baseline`.
- **One GPU, nothing else running.** Check `nvidia-smi` before you start, and
  don't share the GPU with other jobs.

## Output schema

`schema_version` is `1`. A run directory is named
`<UTC timestamp>-<backend>-<model slug>`, for example
`20261001T120000Z-vllm-meta-llama--Llama-3.1-8B-Instruct`. The slug replaces
`/` with `--` and any other character outside `[A-Za-z0-9._-]` with `-`. The
directory contains exactly these three files. Times are in seconds, memory in
bytes, and every percentile uses linear interpolation (numpy's default). In
JSON, unknown or not-applicable values are `null`.

### env.json

One JSON object.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | Output schema version (1) |
| `run_id` | string | The run directory's name |
| `created_utc` | string | ISO 8601 UTC start time |
| `cli_argv` | list of strings | The exact command-line arguments (after `run.py`) |
| `args` | object | Every resolved option (after `--dry-run` and backend defaults), keyed by argparse destination: `backend`, `model`, `num_prompts`, `batch_size`, `max_new_tokens`, `input_len`, `scenarios`, `warmup`, `repeats`, `out`, `dry_run`, `seed`, `probe_delay_ms`, `slow_queue_depth`, `flat_tolerance_pct`, `worker_pool_size`, `device`, `dtype`, `vllm_engine_kwargs`, `isolate` |
| `scenarios` | list of strings | Expanded scenario names, in run order |
| `dry_run` | bool | Whether `--dry-run` was given |
| `device` | string or null | hf: the torch device; vllm: null |
| `git_sha` | string or null | `git rev-parse HEAD` of the checkout holding `run.py` |
| `git_dirty` | bool or null | Whether tracked files had uncommitted changes |
| `python` | string | Python version |
| `platform` | string | `platform.platform()` |
| `cpu` | object | `model` (string or null), `logical_cores` (int) |
| `gpu` | object | `count` (int), `devices` (list of `{index, name, total_memory_bytes}`), `driver_version` (string or null, from `nvidia-smi`), `cuda_version` (string or null, `torch.version.cuda`) |
| `versions` | object | Installed versions (string or null): `undercurrent`, `torch`, `transformers`, `vllm`, `numpy` |
| `failed_scenarios` | list of objects | `{scenario, error}` for each scenario that raised; empty on success |

### results.jsonl

One JSON object per line, one line per (scenario, repeat), in run order.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | Output schema version (1) |
| `run_id` | string | Same as in `env.json` |
| `backend` | string | `hf` or `vllm` |
| `model` | string | Model id or path |
| `scenario` | string | Scenario name, e.g. `inline-8`, `async-slow:drop_oldest` |
| `repeat` | int | 0-based repeat index |
| `num_probes` | int | Extraction points in the scenario |
| `num_inline_probes` | int | Of which inline |
| `num_async_probes` | int | Of which async |
| `probe_impls` | string | Probe kinds used, sorted and joined with `+` (`norm`, `mlp`, `mlp+norm`, `sleep`); empty for `baseline` and `probed-0` |
| `overflow_policy` | string or null | `async-slow` only: `drop_oldest`, `drop_newest` or `block` |
| `queue_depth` | int or null | `async-slow` only: the probe's `queue_depth`; null means the router default |
| `probe_delay_ms` | float or null | `async-slow` only: sleep per activation |
| `worker_pool_size` | int or null | Router worker pool size in force; null for `baseline` |
| `num_prompts` | int | Prompts per repeat |
| `batch_size` | int | Prompts per `generate()` call |
| `max_new_tokens` | int | Tokens generated per prompt |
| `input_len` | int | Requested prompt length in tokens |
| `prompt_tokens_mean` | float | Actual mean prompt length after re-tokenising |
| `prompts_sha256` | string | SHA-256 of the prompt list; equal across all rows of a run |
| `seed` | int | `--seed` |
| `num_batches` | int | `generate()` calls per repeat |
| `wall_time_s` | float | Wall time of the whole repeat |
| `batch_latencies_s` | list of floats | Latency of every batch, in order |
| `latency_p50_s` | float | Median batch latency |
| `latency_p90_s` | float | 90th percentile batch latency |
| `latency_p99_s` | float | 99th percentile batch latency |
| `latency_mean_s` | float | Mean batch latency |
| `output_tokens` | int | Tokens generated in the repeat |
| `output_tokens_counted` | bool | True if counted from the engine; false if inferred as `num_prompts × max_new_tokens` (`ProbedModel` on vLLM returns text only) |
| `output_tokens_per_s` | float | `output_tokens / wall_time_s` |
| `ttft_s` | list of floats or null | Per-request time to first token (hf); null on vllm |
| `ttft_p50_s` | float or null | Median TTFT |
| `ttft_p90_s` | float or null | 90th percentile TTFT |
| `ttft_p99_s` | float or null | 99th percentile TTFT |
| `ttlt_s` | list of floats or null | Per-request time to last token (hf); null on vllm |
| `ttlt_p50_s` | float or null | Median time to last token |
| `peak_gpu_mem_bytes` | int or null | Peak `torch.cuda.max_memory_allocated` in the repeat (hf on CUDA only) |
| `probe_activations` | int | `on_activation` calls the probes completed |
| `probe_compute_s` | float | Total time inside probes' `on_activation` |
| `inline_probe_compute_s` | float | The part of `probe_compute_s` spent in inline probes |
| `queue_drops` | int or null | Activations discarded by the overflow policy (router metrics `record_drop`); null for `baseline` |
| `probe_errors` | int or null | Async `on_activation` calls that raised; null for `baseline` |
| `queue_depth_max` | int or null | Highest total async backlog observed; null for `baseline` |
| `queue_depth_series` | list of `[t_s, depth]` or null | Total async backlog (sum over all live async queues) over time, `t_s` from the start of the repeat; at most 200 points, each the maximum of its time bucket; null for `baseline` |
| `overhead_pct_latency_p50` | float or null | `(latency_p50_s / baseline latency_p50_s − 1) × 100`, against the pooled `baseline` summary; null without a `baseline` scenario |
| `overhead_pct_tokens_per_s` | float or null | `(1 − output_tokens_per_s / baseline output_tokens_per_s_mean) × 100` (positive means slower) |
| `overhead_pct_ex_inline_probe_compute` | float or null | `((wall_time_s − inline_probe_compute_s) / baseline mean wall_time_s − 1) × 100` |
| `latency_flat` | bool or null | `async-slow` only: generation latency (`ttlt_p50_s` if measured, else `latency_p50_s`) within `--flat-tolerance-pct` of the baseline's |

### summary.csv

A header row, then one row per scenario in run order, aggregated over its
repeats. Empty cells mean null. Booleans are `true`/`false`. Floats are
written at full precision.

| Column | Meaning |
| --- | --- |
| `schema_version` | Output schema version (1) |
| `backend` | As in `results.jsonl` |
| `model` | As in `results.jsonl` |
| `scenario` | As in `results.jsonl` |
| `repeats` | Rows aggregated |
| `num_probes` | As in `results.jsonl` |
| `num_inline_probes` | As in `results.jsonl` |
| `num_async_probes` | As in `results.jsonl` |
| `probe_impls` | As in `results.jsonl` |
| `overflow_policy` | As in `results.jsonl` |
| `queue_depth` | As in `results.jsonl` |
| `probe_delay_ms` | As in `results.jsonl` |
| `worker_pool_size` | As in `results.jsonl` |
| `num_prompts` | As in `results.jsonl` |
| `batch_size` | As in `results.jsonl` |
| `max_new_tokens` | As in `results.jsonl` |
| `input_len` | As in `results.jsonl` |
| `latency_p50_s` | Median of all repeats' batch latencies pooled |
| `latency_p90_s` | 90th percentile, pooled |
| `latency_p99_s` | 99th percentile, pooled |
| `latency_mean_s` | Mean, pooled |
| `output_tokens_per_s_mean` | Mean of the repeats' `output_tokens_per_s` |
| `output_tokens_per_s_std` | Sample standard deviation of the same (empty with one repeat) |
| `ttft_p50_s` | Median TTFT, pooled (hf only) |
| `ttft_p90_s` | 90th percentile TTFT, pooled (hf only) |
| `ttft_p99_s` | 99th percentile TTFT, pooled (hf only) |
| `ttlt_p50_s` | Median time to last token, pooled (hf only) |
| `peak_gpu_mem_bytes` | Maximum over repeats |
| `probe_activations_mean` | Mean over repeats |
| `probe_compute_s_mean` | Mean over repeats |
| `inline_probe_compute_s_mean` | Mean over repeats |
| `queue_drops_total` | Sum over repeats |
| `probe_errors_total` | Sum over repeats |
| `queue_depth_max` | Maximum over repeats |
| `overhead_pct_latency_p50` | `(latency_p50_s / baseline latency_p50_s − 1) × 100` |
| `overhead_pct_tokens_per_s` | `(1 − output_tokens_per_s_mean / baseline output_tokens_per_s_mean) × 100` |
| `overhead_pct_ex_inline_probe_compute` | `((mean wall_time_s − inline_probe_compute_s_mean) / baseline mean wall_time_s − 1) × 100` |
| `latency_flat` | `async-slow` only, as in `results.jsonl`, from the pooled values |

Changing a field's name, type or meaning means bumping `SCHEMA_VERSION` in
`run.py` and updating this section. `tests/benchmarks/test_harness.py` checks
that the files match these tables.

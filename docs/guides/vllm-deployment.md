# Deploy with vLLM

<!-- owner: p3-vllm-guide -->

This page covers running Undercurrent on [vLLM](https://docs.vllm.ai/):
installing it next to the vLLM you already run, generating with
`ProbedModel(..., backend="vllm")`, returning probe verdicts over HTTP, and
which vLLM versions are supported. If you have your own request loop or
server and want to drive the engine adapter and `Router` yourself, read
[Embed in your serving stack](../production/embedding.md) after this page.

!!! info "Requires a GPU"
    Every vLLM example on this page needs a CUDA GPU and a working vLLM
    install. None of them run in the docs CI. To try Undercurrent without a
    GPU, use the default Hugging Face backend from the
    [Quickstart](../getting-started/quickstart.md).

!!! warning "Known issues (v0.1)"
    The vLLM adapter hasn't been verified on a GPU yet. Fixed in 0.1.0: on
    Llama-family models (and other fused-residual architectures),
    `residual_stream` is captured as `hidden_states + residual`, the same
    residual stream as the HF backend; earlier builds captured the layer's
    MLP output. GPU verification of the fix is pending. See
    [Known issues](../compatibility.md#known-issues-v01) for details.

## Install into your vLLM environment

vLLM ships builds matched to a specific torch and CUDA version. Undercurrent
doesn't try to pick those for you. The recommended production setup is to
install Undercurrent **into an environment or image that already has a
working, GPU-matched vLLM**:

```bash
# in the virtualenv, conda env or container where vLLM already works
pip install undercurrent
```

For your own container image, start from the vLLM image you already use and
add Undercurrent on top:

```dockerfile
# Pick a vllm/vllm-openai tag whose vLLM is in Undercurrent's supported range
# (see docs/compatibility.md).
FROM vllm/vllm-openai:<tag>

RUN pip install undercurrent

# The base image's entrypoint starts vLLM's own OpenAI-compatible server, which
# doesn't drive probes (see "Serving over HTTP" below). Run your own server.
COPY server.py /app/server.py
ENTRYPOINT ["python3", "/app/server.py"]
```

The snippet is a starting point for your own image; adapt the tag, the
entrypoint and how you install your server's code.

**Undercurrent never installs or upgrades vLLM or torch behind your back.**
vLLM isn't a dependency of `pip install undercurrent` at all, and the torch
requirement is a loose floor (`torch>=2.1`, no upper bound), so it never
fights the exact torch version your vLLM pins. Installing Undercurrent into a
working vLLM environment leaves vLLM and torch as they were. The project's
GPU CI installs vLLM first, then Undercurrent, and fails if either version
changed.

**The convenience extra.** If you're starting from an empty environment, for
example on a development GPU box, you can let pip install a vLLM from the
tested range:

```bash
pip install "undercurrent[vllm]"
```

This resolves `vllm>=0.28,<0.29` together with whatever torch that vLLM
wheel requires. Use it when you don't already have a vLLM to match. For
production images, prefer the vLLM install (or base image) you already trust,
and add `pip install undercurrent` to it.

**The runtime version check.** Because pip never sees the supported range
when you install into an existing environment, the adapter checks it itself.
Constructing a `VLLMEngineAdapter`, which `ProbedModel(backend="vllm")` does
for you, reads the installed vLLM version from package metadata (it doesn't
import vLLM) and compares it with `SUPPORTED_VLLM` (`>=0.28,<0.29`). Outside
that range it raises `VLLMAdapterLimitationError` before any model is
loaded; see [Troubleshooting](#troubleshooting). If vLLM isn't installed at
all, you get `MissingDependencyError` instead, listing the same two install
routes.

[Compatibility](../compatibility.md) lists the declared ranges, the
combinations that have actually been tested, and how to override the check.

## Generate with `ProbedModel`

`ProbedModel` is the same front door as in the
[Quickstart](../getting-started/quickstart.md); pass `backend="vllm"`.
Keyword arguments that `ProbedModel` doesn't use itself go to vLLM as engine
arguments (`gpu_memory_utilization`, `max_model_len`, `dtype`,
`trust_remote_code`, ...), together with the adapter's own switches such as
`allow_unsupported_executor`.

!!! info "Requires a GPU"
    This example needs a CUDA GPU and vLLM.

```py
from undercurrent.core import ActivationRecord, probe
from undercurrent.model import ProbedModel

SPEC = """
version: "1"
extraction_points:
  - name: last_prompt_token
    layers: 6
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: norm_gate
    probe_kind: single_shot
    execution_mode: inline
"""


@probe("norm_gate", threshold=500.0)
def norm_gate(record: ActivationRecord) -> float:
    return float(record.tensor.float().norm())


prompts = ["Hello, my name is", "The capital of France is", "Once upon a time"]

with ProbedModel.from_pretrained(
    "openai-community/gpt2",
    backend="vllm",
    spec=SPEC,
    probes={"norm_gate": norm_gate},
    gpu_memory_utilization=0.3,  # vLLM engine args from here on
    max_model_len=1024,
    enforce_eager=True,  # needed for capture; see "Scaling and limits"
) as model:
    outputs = model.generate(prompts, max_new_tokens=32, temperature=0)

for out in outputs:
    print(out.request_id, repr(out.text), out.probe_results["last_prompt_token"].verdict)
```

What's different from the HF backend:

- **Batching.** `generate()` on a list runs up to `max_concurrency` prompts
  at once (default 64, `DEFAULT_VLLM_MAX_CONCURRENCY`; set it with
  `ProbedModel(..., max_concurrency=...)`), and vLLM's scheduler batches them.
  The outputs come back as a list in prompt order. Each prompt is its own
  request with its own `request_id` and its own probe instances, so results
  never mix between prompts.
- **Failures in a batch.** By default the first failing prompt raises for the
  whole call, and prompts that haven't started are cancelled. Pass
  `return_exceptions=True` to get the exception in that prompt's slot of the
  returned list instead.
- **Generation arguments.** `max_new_tokens` becomes vLLM's `max_tokens`
  (default 32 when you don't give one), `min_new_tokens` becomes
  `min_tokens`, and `temperature` (0 is greedy), `top_p`, `seed` and `stop`
  map one to one onto `SamplingParams`. Any other keyword argument is passed
  through as a `SamplingParams` field.
- **Device.** vLLM places the model on its GPU(s) itself. Leave `device=`
  unset; anything other than `None` or a `cuda` device raises
  `ProbedModelConfigError`.
- **Tensor types.** `kv` isn't supported on vLLM. `ProbedModel` rejects it,
  and layer indices that are out of range for the model, at construction,
  before the model loads.

### Stream results into your own system: `on_result=`

`on_result` is called once per finished generation with its
`GenerationOutput`, as soon as that prompt finishes, not when the whole batch
is done. With the vLLM backend it runs **on the thread that finished the
generation**, one of the batch's worker threads, so several calls can run at
the same time. Keep the callback thread-safe and quick, for example by
putting the output on a queue that another thread ships to your database or
message bus. If the callback raises, the exception is logged on the
`undercurrent` logger and the generation's output is still returned.

!!! info "Requires a GPU"
    This example needs a CUDA GPU and vLLM.

```py
import queue

results: queue.Queue = queue.Queue()

with ProbedModel.from_pretrained(
    "openai-community/gpt2",
    backend="vllm",
    spec=SPEC,
    probes={"norm_gate": norm_gate},
    on_result=results.put,  # queue.Queue.put is thread-safe
    enforce_eager=True,
) as model:
    model.generate(prompts, max_new_tokens=32)

while not results.empty():
    out = results.get()
    print(out.request_id, out.aborted, out.probe_results["last_prompt_token"].verdict)
```

For durable, out-of-process delivery of async (observe-mode) probe output,
use a log sink instead (`ProbedModel(..., log_sink=...)`); see
[Observation sinks](observation-sinks.md).

### What an inline abort looks like

When an inline probe returns an `ABORT` signal, the adapter cancels the
request through vLLM's own abort API. The generation's `GenerationOutput`
then has:

| Field | Value after an abort |
| --- | --- |
| `aborted` | `True` |
| `abort_point` | Name of the extraction point that stopped generation, for example `"last_prompt_token"` |
| `abort_signal` | The `ProbeSignal` that stopped it (its `confidence` and `metadata`) |
| `abort_reason` | A readable summary, for example `extraction point 'last_prompt_token' aborted generation (confidence 0.97)` |
| `text` | The text generated **before vLLM stopped the request** |
| `probe_results` | Every extraction point's `ProbeResult`, the aborting one included |

Two things differ from the HF backend:

- **The abort lands a step late.** vLLM schedules a whole step before it
  checks for aborted requests, so a request stops at the start of the next
  scheduler step. Expect at least one extra token after the one whose
  activation triggered the abort, and a few more under heavy concurrent load.
  Don't return `text` from an aborted generation to your users if the abort
  is a safety gate.
- **`aborted` comes from the probe's results.** The vLLM adapter doesn't
  report which extraction point stopped a request, so `ProbedModel` finds the
  earliest `ABORT` signal in the inline extraction points'
  `ProbeResult.signal_history`. Function probes (`@probe`) record their
  signals there automatically. A class probe must include the `ABORT` signals
  it returned in the `signal_history` of the `ProbeResult` from `on_end`, as
  the [Quickstart's](../getting-started/quickstart.md#5-stop-a-generation)
  `MeanNormProbe` does. Otherwise vLLM still stops the request, but
  `aborted` is `False`.

```py
for out in outputs:
    if out.aborted:
        print(f"{out.request_id}: stopped by {out.abort_point}: {out.abort_reason}")
    else:
        print(f"{out.request_id}: {out.text!r}")
```

### Bring your own adapter or router

Everything below `ProbedModel` stays reachable:

- `router_kwargs={...}` is forwarded to `Router(...)`: `worker_pool_size`,
  `default_queue_depth`, `default_overflow_policy`, `drain_timeout`,
  `default_intervention_policy`, `circuit_breaker_threshold`. See
  [Embed in your serving stack](../production/embedding.md#tune-for-production)
  for what to set.
- `log_sink=` and `metrics_sink=` attach an observation sink and a metrics
  sink.
- `backend=VLLMEngineAdapter()` uses an adapter you built and loaded
  yourself (pass `None` as the model). You keep owning it: `close()` doesn't
  shut it down.
- `model.router` and `model.adapter` expose the live objects.

## Serving over HTTP

Undercurrent v0.1 doesn't include an HTTP server. A supported, OpenAI-compatible
`serve` command is [on the roadmap](https://github.com/wrynx/undercurrent/blob/main/ROADMAP.md).
Until then, put `VLLMEngineAdapter` (or `ProbedModel`) behind the web server
you already run.

[`examples/openai_server/`](https://github.com/wrynx/undercurrent/tree/main/examples/openai_server)
is the reference to copy. It isn't installed with the package. It has two
parts:

- **`serve_llama_mlp_pipeline.py`**: a working HTTP server. It builds one
  `Router` and one `VLLMEngineAdapter` at startup, shares them across
  requests, and handles each HTTP request on its own thread with
  `register_extraction` → `generate` → `unregister_extraction`. It collects
  each request's probe results with `router.on_request_end`. It uses the
  stdlib `ThreadingHTTPServer`, which isn't hardened for production (no
  backpressure, request timeouts or TLS); keep its `run_generate()` logic and
  put it behind uvicorn, gunicorn or your own framework.
- **`response_schema.py`**: a pure formatting layer, with no HTTP or vLLM
  dependency, that turns a finished generation and its probe verdicts into
  OpenAI-style response bodies and SSE chunks.

To run the reference server (it needs a probe checkpoint; the
[README](https://github.com/wrynx/undercurrent/tree/main/examples/openai_server)
and the script's docstring cover making a placeholder one):

```bash
export PROBE_MODEL_PATH=/path/to/checkpoint.pt
python examples/openai_server/serve_llama_mlp_pipeline.py --port 8000
```

```bash
curl -s localhost:8000/generate \
  -d '{"prompt": "Tell me about the ocean.", "max_tokens": 64}'
```

The server listens on `127.0.0.1` by default. In a container, pass
`--host 0.0.0.0` so the published port can reach it. The server has no
authentication, so keep it off untrusted networks.

The reference server returns its own simple shape: `request_id`, `prompt`,
`text`, and a `verdicts` object with each extraction point's
`ProbeResult.verdict`. To return the OpenAI-style wire format instead, build
the body with `response_schema.build_completion_response(outcome, verdicts)`
from a `GenerationOutcome` and one `ProbeVerdict` per extraction point.

A clean completion carries a `probing` object with one entry per extraction
point, inline and async alike:

```json
{
  "flagged": false,
  "finish_reason": "stop",
  "text": "hello world",
  "tokens_generated": 42,
  "max_tokens": 256,
  "probing": {
    "toxicity_probe": {
      "score": 0.12,
      "flagged": false,
      "execution_mode": "inline",
      "evaluated": true
    },
    "trajectory_probe": {
      "score": {"max_score": 0.4, "final_score": 0.2},
      "flagged": false,
      "execution_mode": "async",
      "evaluated": true
    }
  }
}
```

- `score` is a float for a single-shot probe, or `{max_score, final_score}`
  for a trajectory probe (`aggregate_trajectory_scores`).
- `evaluated` is `false` when the extraction point's position was never
  reached, for example a `generated[*]` point on a generation that produced
  no tokens. Then `score` is `null`.

When an inline probe aborted the generation, the body has a different shape.
The text is suppressed and `finish_reason` is `"content_flagged"` instead of
`"stop"`:

```json
{
  "flagged": true,
  "finish_reason": "content_flagged",
  "text": null,
  "flagged_by": "toxicity_probe",
  "flagged_score": 0.94,
  "tokens_generated": 17,
  "max_tokens": 256,
  "tokens_saved": 239
}
```

- `flagged_by` and `flagged_score` come from the first inline, evaluated,
  flagged verdict. An async probe is never reported as the cause, because
  async probes never stop generation.
- `tokens_saved` is `max_tokens - tokens_generated`.

For streaming, `build_flagged_sse_chunk()` builds the terminal chunk of a
flagged stream (`choices[0].finish_reason == "content_flagged"`, empty
`delta`), and `build_probing_result_sse_event()` builds an extra
`probing_result` event to send after a clean stream's normal final chunk.

## Scaling and limits

- **One worker per engine.** The adapter is validated with a single vLLM
  worker. If vLLM starts more than one worker, for tensor or pipeline
  parallelism, `load_model()` raises `VLLMAdapterLimitationError` unless you
  pass `allow_unsupported_executor=True`. With the override, the adapter
  merges every worker's records and drops tensor-parallel duplicates, but
  multi-GPU capture isn't validated yet. To scale out today, run several
  single-GPU engines behind a load balancer.
  [Tensor & pipeline parallelism](../internals/vllm-parallelism.md) has the
  per-topology support status.
- **Use `enforce_eager=True`.** With CUDA graphs, replayed steps don't run
  Python, so the forward hooks can't fire. The adapter doesn't set it for you.
- **`block_until_signal` stalls the whole batch.** Under continuous batching,
  a blocking inline probe holds the scheduler step that every co-batched
  request shares, for up to `timeout_ms`. The adapter logs a one-time warning
  when it sees such an extraction point. Use `block_until_signal` on vLLM only
  for single-request or offline runs; with concurrent traffic, keep inline
  probes cheap with the default `mode: reject`, or run them
  `execution_mode: async`. See
  [Intervention policies](../production/intervention-policies.md#recommended-production-settings).
- **Model coverage.** Some captures silently produce nothing or something
  other than you'd expect on particular architectures (for example `attn_out`
  on GPT-2-style models, or `residual_stream` on Llama-family layers). Check
  [Known limitations](../internals/vllm-adapter.md#known-limitations) for your
  model before relying on a probe.
- **`vllm serve`.** Undercurrent registers a `vllm.general_plugins` entry
  point, but a bare `vllm serve` process doesn't drive probing. See
  [Running under `vllm serve`](../internals/vllm-adapter.md#running-under-vllm-serve).

[The vLLM adapter](../internals/vllm-adapter.md) explains how capture,
position mapping and aborts work.

## Supported versions

The supported vLLM range is a single minor release, currently
`vllm>=0.28,<0.29`. The adapter hooks undocumented vLLM internals that can
change in any minor release, so the range stops at the next one.

- **Pinned minor.** This is the range the GPU test suite targets. Before
  every release a maintainer runs it on a GPU machine with
  `scripts/gpu_check.sh`, installed the documented way: vLLM first, then
  Undercurrent into the same environment.
- **Moving to a new minor.** The range is widened, in the `[vllm]` extra,
  `SUPPORTED_VLLM` and [Compatibility](../compatibility.md) together, only
  after the GPU suite passes on the new minor. The previous minor stays
  best-effort for one more Undercurrent release, then it's dropped.

!!! warning "GPU CI status"
    There is no GPU CI for v0.1: the GPU tests run by hand before each release
    (`scripts/gpu_check.sh`). The vLLM rows on the
    [Compatibility](../compatibility.md#tested-combinations) page say which
    combinations have had a GPU run.

## Troubleshooting

**`VLLMAdapterLimitationError: undercurrent.adapters.vllm supports vllm>=0.28,<0.29, but vllm==X is installed. ...`**

Raised when the adapter is constructed. Your vLLM is outside the supported
range. Either use an environment or base image with a vLLM in range, or
install the tested range with `pip install "undercurrent[vllm]"` in a fresh
environment. To try an untested version anyway, for example to evaluate a
new vLLM release, set `UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM=1` (`true`,
`yes` and `on` also work). The error becomes a `RuntimeWarning`, and the
checks below still fail loudly if vLLM's internals changed.

**`MissingDependencyError: backend='vllm' needs vLLM, which undercurrent never installs by default. ...`**

vLLM isn't importable in this environment. Install Undercurrent into your
vLLM environment, or use `pip install "undercurrent[vllm]"`.

**`VLLMAdapterLimitationError: detected N workers (via collective_rpc's one-result-per-worker fan-out). ...`**

Your engine runs more than one worker, usually because
`tensor_parallel_size` or `pipeline_parallel_size` is above 1. Run one
worker per engine, or pass `allow_unsupported_executor=True` if you accept
an unvalidated topology (see [Scaling and limits](#scaling-and-limits)).

**`VLLMAdapterLimitationError: could not determine worker topology (a probe RPC call raised ...)`**

The adapter's worker-count check failed. This usually means the installed
vLLM's RPC surface differs from what the adapter expects. Check the vLLM
version against [Compatibility](../compatibility.md). If you're certain the
engine has exactly one worker, `allow_unsupported_executor=True` skips the
check.

**`VLLMAdapterLimitationError: could not find a collective_rpc() entry point on the loaded engine ...`** or **`... could not locate a tokenizer on the loaded engine ...`**

The engine object doesn't look like the vLLM the adapter was written
against. This happens with an unsupported vLLM version, typically one
loaded with the override. Use a [supported version](../compatibility.md).

**`VLLMAdapterLimitationError: a different Router instance was already bound to this adapter ...`**

One `VLLMEngineAdapter` serves exactly one `Router` for its whole lifetime.
Build one `Router` at startup and pass the same instance to every
`generate()` call.

**`VLLMIntrospectionError: model_runner.input_batch is missing ...`**, **`... is missing expected attribute(s) [...]`**

Raised on the first forward hook when vLLM's internal batch layout isn't
the V1 model-runner layout the adapter reads. Causes, most likely first:
an unsupported vLLM version (with the version check overridden), or
`allow_v2_model_runner=True` (the adapter only understands the V1 model
runner; keep the default, which forces V1). The adapter fails here on
purpose rather than attach activations to the wrong tokens.

**`SeqMapperError: ... has a scheduled row range but was never registered via register_request()`**

vLLM reported a request id the adapter didn't register. This happens when
vLLM rewrites request ids, which the adapter prevents by setting
`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` in `load_model()`. You see this
error if you passed `allow_request_id_randomization=True`, or if your vLLM
no longer honours that variable (an unsupported version).

**`WorkerExtractionError: could not locate a decoder-layer list on the loaded model ...`**

The model's architecture keeps its decoder layers somewhere the adapter
doesn't look (it checks `model.model.layers`, `model.layers` and
`model.transformer.h`). Use a supported architecture, or add the path in
`ProbingWorkerExtension._find_decoder_layers()` and open a pull request.

**No probe results, or `aborted` is `False` although a probe asked to abort**

Check that you passed `enforce_eager=True`, that the extraction point's
`tensor_type` and `position` are captured on your model (see
[Known limitations](../internals/vllm-adapter.md#known-limitations)), and
that a class probe puts its `ABORT` signals in `signal_history` (see
[What an inline abort looks like](#what-an-inline-abort-looks-like)).

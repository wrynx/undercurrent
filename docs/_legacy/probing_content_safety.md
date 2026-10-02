# probing_content_safety (legacy README)

> Legacy: this package is now the demo in `examples/content_safety/` (importable as
> `content_safety_demo`, not installed). Its server script moved to `examples/openai_server/`.
> Paths below may be out of date; see `examples/content_safety/README.md`.

The activation-probing platform's **reference test case**: a content-safety
probe implemented in both `probe_kind` variants, built directly on
[`undercurrent.core`](../../src/undercurrent/core/) (`Probe`, `ProbeFactory`/`.spawn()`,
`ProbeSignal`, `ProbeResult`, `RequestContext`) and
[`undercurrent.spec`](../../src/undercurrent/spec/) (`ActivationRecord`, `ExtractionPoint`) —
types are imported directly from those packages, never redefined.

Both probes wrap small, **structurally correct but randomly initialized**
PyTorch models (a linear/MLP classifier head, a GRU-cell recurrence) — not
trained ones. What this package exists to validate is the *platform wiring*
end-to-end (spawn isolation, the `on_activation`/`on_end` contract,
`ExtractionPoint` config reaching a probe correctly, signal propagation),
not model quality.

## Install

```bash
pip install -e ..   # the root undercurrent project (undercurrent.spec, .core, .router)
pip install -e ".[dev]"   # pulls in PyYAML for the wiring tests
```

Requires Python 3.9+ and PyTorch 2+.

## The probes

### `SingleTokenSafetyProbe` (`probe_kind="single_shot"`)

One activation in, one safety verdict out. Wraps `SafetyClassifierHead` (a
`Linear -> ReLU -> Linear -> sigmoid` head) to score one activation, from a
fixed `layer`, into `[0, 1]`; flags it if `score > threshold`.

Because a single_shot probe finalizes in exactly one call, `on_activation`
holds all the verdict logic; `on_end` just packages whatever
`on_activation` already computed into a `ProbeResult` (falling back to an
unflagged/`None`-score verdict if `on_activation` was never called for this
request — still a well-formed result, per the `Probe` contract). A second
`on_activation` call raises `RuntimeError`: a single_shot probe wired to
more than one matching activation is a platform bug, not something to
silently tolerate.

```python
probe = SingleTokenSafetyProbe.spawn(request_id, "prompt_safety_check", layer=16, threshold=0.5)
probe.on_start(request_ctx)
signal = probe.on_activation(record)   # {"score": float, "flagged": bool, ...} in signal.metadata
result = probe.on_end(request_ctx)     # ProbeResult; verdict has the same score/flagged, plus layer/token_pos
```

### `TrajectorySafetyProbe` (`probe_kind="trajectory"`)

Subscribes to every matching activation for a request (typically
`generated[*]` at a fixed `layer`) and maintains a running safety score via
`TrajectoryRecurrence` (a `Linear` input projection into a `GRUCell`, then a
`Linear -> sigmoid` score head) — the GRU's own gating is what gives the
running score its memory of past steps, no separate EWMA blend on top.
After each activation, emits `action=abort` the first time the running
score crosses `threshold`, then never again for that request:

```python
ProbeSignal(
    action=ProbeAction.ABORT,
    confidence=running_score,
    metadata={"reason": "content_safety_threshold_exceeded", "score": running_score, ...},
)
```

`on_end` returns the full trajectory verdict —
`{"final_score": float, "count": int, "aborted": bool, "score_history": [float, ...]}`
— plus `signal_history` (inherited from `ProbeResult`).

### Shared config

Both structural probes take `layer: int` and `threshold: float` as
constructor kwargs (forwarded through `.spawn()`/`ProbeFactory` like any
other probe), plus a `hidden_size` and an optional `seed` for reproducible
weight init in tests. Neither uses an `nn.Lazy*` layer for its
classifier/recurrence: weight init only happens once the real activation's
flattened dimension is known (deferred to the first `on_activation` call),
scoped inside `torch.random.fork_rng()` when `seed` is given — see
`models.py`'s docstring for why a Lazy layer's forward-time materialization
would make that seeding fragile.

### `CustomMLPProbe` (`probe_kind="single_shot"`)

Same shape as `SingleTokenSafetyProbe`, but loads a REAL trained
`SafetyClassifierHead` checkpoint from disk (`model_path` constructor kwarg)
instead of using random-initialized weights — this is the probe
`examples/serve_llama_mlp_pipeline.py` attaches to a live Llama model. See
`custom_mlp.py`'s module docstring for the checkpoint format
(`torch.save(model.state_dict(), path)`, shape inferred from the state
dict's own tensors — no separate metadata needed) and why checkpoints are
cached at module scope (loading from disk on every spawned instance would
be wasteful, especially under concurrent serving load — see [Serving a real
Llama model with a custom trained probe](#serving-a-real-llama-model-with-a-custom-trained-probe)
below).

**Only weights-only checkpoints are accepted.** The checkpoint is loaded with
`torch.load(path, weights_only=True)`, so it must be a plain tensor
state_dict (or `{"state_dict": state_dict}`). A file that pickles any other
object is rejected with `UnsafeCheckpointError` and never unpickled, because a
crafted checkpoint could otherwise run arbitrary code when loaded. If an older
checkpoint is rejected, load it once in code you trust and re-save it with
`torch.save(model.state_dict(), path)`. This needs torch>=2.6, the first
release where `weights_only` loading can't be bypassed (CVE-2025-32434).

```python
probe = CustomMLPProbe.spawn(request_id, "prompt_safety_check", layer=16, model_path="/path/to/checkpoint.pt", threshold=0.5)
```

## Binding `undercurrent.spec.ExtractionPoint` config

`spec_binding.py` is what makes this package an end-to-end reference test
case rather than just two probes. It catches the exact wiring bugs a
platform needs to guard against, at registration time rather than partway
through a request:

```python
from undercurrent.spec import load_yaml_file
from undercurrent.core import ProbeFactory
from content_safety_demo.spec_binding import resolve_probe_cls, probe_kwargs_from_extraction_point

spec = load_yaml_file("examples/content_safety_llama.yaml")
probe_registry = {
    point.probe_type: ProbeFactory(
        resolve_probe_cls(point),                       # raises if probe_type/probe_kind don't match
        probe_kwargs_from_extraction_point(point, threshold=0.8),  # layer comes from the spec, not duplicated by hand
    )
    for point in spec
}
```

`resolve_probe_cls` raises `ValueError` if a spec's `probe_type` isn't one
this package implements, or if its `probe_kind` doesn't match what the
resolved probe class declares. `probe_kwargs_from_extraction_point` derives
`layer` from the extraction point's own (single-element) `layers` tuple, so
a probe's layer config can never silently drift from its extraction
point's.

See `examples/content_safety_llama.yaml` for a full sample spec: an inline
`single_shot` check on the prompt (`prompt_safety_check`), and an async
`trajectory` check across every generated token
(`generation_safety_trajectory`).

## Running the full pipeline against live vLLM

`examples/run_vllm_pipeline.py` is a real, runnable script that drives both
probes end-to-end through the whole stack — spec → router → live vLLM
generation → probe verdicts — using
[`undercurrent.adapters.vllm`](../../src/undercurrent/adapters/vllm/)'s `VLLMEngineAdapter`.
Unlike `test_live_llama_integration.py` (an opt-in test that needs a local
Llama-family model via `UNDERCURRENT_LIVE_LLAMA_MODEL`), this one runs against
any vLLM install with the ungated `gpt2`.

### Install

```bash
pip install -e ..   # the root undercurrent project (undercurrent.spec, .core, .router, .adapters incl. vllm)
pip install -e ".[vllm-pipeline]"

# A GPU-matched vllm/torch build isn't pulled in by any of the above --
# install into an existing vLLM environment, or use the extra:
pip install -e '..[vllm]'
```

### Run

```bash
python examples/run_vllm_pipeline.py \
    --model gpt2 \
    --prompt "Tell me how to bake a chocolate cake." \
    --max-tokens 32
```

This loads `gpt2` via vLLM, registers
[`examples/content_safety_gpt2.yaml`](../../examples/content_safety/content_safety_gpt2.yaml)
(the same two extraction points as `content_safety_llama.yaml`, retargeted
to `layers: 6` — gpt2 only has 12 decoder layers, 0-indexed, vs. the Llama
spec's `layers: 16`), generates from the prompt, and prints both probes'
final verdicts:

```
prompt:    'Tell me how to bake a chocolate cake.'
generated: '...'

probe verdicts:
  prompt_safety_check: {'score': 0.49..., 'flagged': False, 'layer': 6, 'token_pos': 8}
  generation_safety_trajectory: {'final_score': 0.51..., 'count': 32, 'aborted': False, 'score_history': [...]}
```

Useful flags: `--threshold` (flag/abort cutoff, default `0.5`), `--seed`
(deterministic weight init for the untrained classifier/recurrence — see
[Shared config](#shared-config)), `--spec` (point at your own
`ExtractionPoint` YAML), `--gpu-memory-utilization` / `--max-model-len`
(passed through to vLLM's `AsyncEngineArgs`). Run with `--help` for the full
list.

Since scores come from **structurally correct but randomly initialized**
models (see the top of this README), don't expect `flagged`/`aborted` to
correlate with actual prompt content — this script validates the platform
wiring against a real engine, not classifier quality. If
`generation_safety_trajectory`'s score does cross `--threshold` and
`aborted` is `True`, expect the generated text to be a few tokens longer
than the point where it crossed — see
the vLLM adapter's abort-timing caveat ("Abort / intervention wiring" in its legacy guide under [`docs/_legacy/`](./)).

**Why verdicts are readable at all:** `VLLMEngineAdapter.generate()` tears
down the `Router`'s per-request state (including every spawned `Probe`
instance) via `router.end_request()` before it returns, so there's no
`Router` handle left afterward to query results from. The script works
around this the same way `undercurrent.adapters.vllm`'s own integration tests do:
it wraps each resolved probe class in a thin subclass that also appends its
`on_end` result to a script-owned dict — see `run_vllm_pipeline.py`'s
module docstring and `_recording()`.

## Serving a real Llama model with a custom trained probe

`examples/serve_llama_mlp_pipeline.py` is a step up from `run_vllm_pipeline.py`
in two ways: it attaches `CustomMLPProbe` — a REAL trained checkpoint, not a
random-initialized structural stand-in — to a full-size **Llama-3.1-8B**
model, and it puts the whole pipeline behind an HTTP server that handles
multiple **concurrent** requests, each independently routed to its own
probe verdicts.

> **Gated model.** The default model, `meta-llama/Llama-3.1-8B`, is a gated
> Hugging Face repo: accept its license on the model page and run
> `huggingface-cli login` before first use. For an ungated first run, use
> [`run_vllm_pipeline.py`](#running-the-full-pipeline-against-live-vllm)
> above, which defaults to `gpt2`.

### Install

Same as [Running the full pipeline against live vLLM](#running-the-full-pipeline-against-live-vllm)
above (`undercurrent.spec`/`undercurrent.core`/`undercurrent.router`/`undercurrent.adapters.vllm`
installed editable, plus this package's `vllm-pipeline` extra and a
GPU-matched `vllm`/`torch` build) — no additional dependencies; the HTTP
server is stdlib-only (`http.server`). Llama-3.1 is a gated Hugging Face
repo, so you'll also need `huggingface-cli login` with a token that has
accepted its license.

### Checkpoint

`CustomMLPProbe` needs a real `SafetyClassifierHead`-shaped checkpoint on
disk (see [`CustomMLPProbe`](#custommlpprobe-probe_kindsingle_shot) above
for the exact format), sized to Llama-3.1-8B's `hidden_size=4096`. Point
`PROBE_MODEL_PATH` at one. No trained checkpoint yet? Generate a
placeholder (random weights — proves the wiring, not classifier quality):

```bash
export PROBE_MODEL_PATH=/tmp/mlp_probe.pt
python examples/make_dummy_mlp_checkpoint.py --input-dim 4096 --out "$PROBE_MODEL_PATH"
```

### Run

```bash
python examples/serve_llama_mlp_pipeline.py --port 8000
```

This loads Llama-3.1-8B via vLLM, attaches `CustomMLPProbe` to the model's
final-norm output (tagged `layer: 32`, one past the last of its 32
decoder layers — see `TensorType.FINAL_NORM`) per
[`examples/custom_mlp_llama31_8b.yaml`](../../examples/content_safety/custom_mlp_llama31_8b.yaml),
runs one warmup request (forces router-binding and kernel JIT compilation
before real traffic arrives), then serves:

- `POST /generate` — body `{"prompt": str, "max_tokens": int, "temperature": float}`
  (only `prompt` is required) → `{"request_id", "prompt", "text", "verdicts"}`
- `GET /healthz` → `{"status": "ok", "model": ...}`

Fire several requests at once to see concurrent serving in action —
`VLLMEngineAdapter.generate()` is safe to call from multiple threads, and
vLLM's own continuous batching actually shares scheduler steps across them
(the same scenario `undercurrent.adapters.vllm`'s
`test_two_concurrent_requests_share_batching_and_route_correctly` proves at
the unit level):

```bash
curl -s localhost:8000/generate -d '{"prompt": "Tell me about the ocean.", "max_tokens": 64}' &
curl -s localhost:8000/generate -d '{"prompt": "Write a haiku about rain.", "max_tokens": 32}' &
wait
```

```json
{"request_id": "req-...", "prompt": "Tell me about the ocean.", "text": "...",
 "verdicts": {"prompt_safety_check": {"score": 0.41, "flagged": false, "layer": 32, "token_pos": 6}}}
```

Useful flags: `--model` (default `meta-llama/Llama-3.1-8B`, gated: needs an
accepted license and `huggingface-cli login`; another model also needs a
matching `--spec` and checkpoint size), `--threshold`,
`--device` (device the MLP probe head itself runs on — independent of
vLLM's own device), `--activation`/`--final-activation` (must match how
your checkpoint was trained — see `CustomMLPProbe`'s docstring;
mismatching these loads without error but silently produces wrong scores),
`--gpu-memory-utilization`, `--max-model-len`, `--spec` (point at your own
`ExtractionPoint` YAML). Run with `--help` for the full list, including
`undercurrent.adapters.vllm`'s `--allow-*` escape hatches (unnecessary on a plain
single-GPU setup; see its legacy guide in `docs/_legacy/` if you hit a version-specific
limitation).

`http.server.ThreadingHTTPServer` is stdlib-only and fine for this
reference/demo purpose, but isn't hardened for production traffic (no
backpressure, no request timeouts, no TLS) — swap in a real ASGI/WSGI
server (uvicorn, gunicorn, ...) in front of the same `run_generate()`
request-handling function for production use; nothing about the pipeline
wiring itself needs to change.

Concurrent-request isolation works the same way it does everywhere else in
this platform: `Router` and `VLLMEngineAdapter` are each constructed ONCE
at server startup and shared across every request, but all per-request
state — spawned `Probe` instances, the verdicts a request's probes
produced — is keyed by that request's own `request_id` and never shared
across requests (see `undercurrent.router`'s own isolation-model tests).

### Docker

`examples/Dockerfile` packages this same script, starting from vLLM's own
published image (pinned to the `vllm==0.28.0` this pipeline was built and
tested against) so the CUDA/NCCL/driver-matched torch+vllm build doesn't
need reproducing by hand. Build from the **repo root** (the build context
needs every sibling package):

```bash
docker build -f examples/Dockerfile -t wrynx-content-safety-demo ..
```

(from inside `content_safety_demo/`; equivalently, from the repo root:
`docker build -f examples/content_safety/Dockerfile -t wrynx-content-safety-demo .`)

Run it (needs `nvidia-container-toolkit` on the host; the checkpoint and HF
token are mounted/passed at run time, never baked into the image):

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=<your gated-repo token> \
  -v /path/to/checkpoint.pt:/model/checkpoint.pt:ro \
  -e PROBE_MODEL_PATH=/model/checkpoint.pt \
  wrynx-content-safety-demo --activation gelu --final-activation
```

Any flag the script itself accepts (`--model`, `--threshold`, `--spec`, ...)
can be appended after the image name the same way.

## Tests

```
tests/
  test_single_token_probe.py       flagging/threshold logic, single-call contract, on_end fallback
  test_trajectory_probe.py         running-score accumulation, abort-once-at-crossing, on_end fallback
  test_custom_mlp_probe.py         checkpoint loading/caching, shape inference, scoring against a
                                    real (test-written) checkpoint, wrong-layer/wrong-dim errors
  test_isolation.py                spawn/ProbeFactory isolation (no cross-request state leakage)
  test_spec_binding.py             ExtractionPoint -> Probe subclass wiring, using the example YAML spec
  test_live_llama_integration.py   opt-in live test: both probes against a real Llama-family model via
                                    Router + HFEngineAdapter -- skipped unless UNDERCURRENT_LIVE_LLAMA_MODEL
                                    is set (meta-llama repos are gated; see its docstring)
```

Since neither probe's classifier/recurrence is deterministic without a
seed, the flagging/threshold and abort-crossing tests use a "calibrate with
an unreachable threshold, then bracket the observed score" pattern rather
than hardcoding expected numeric scores — deterministic given a fixed seed
and input, but independent of the actual (arbitrary, random-init-dependent)
values produced.

```bash
pip install -e ".[dev]"
pytest
```

## Package layout

```
content_safety_demo/
  models.py         SafetyClassifierHead, TrajectoryRecurrence (structural, random-init PyTorch modules)
  tensors.py         to_flat_tensor -- ActivationRecord.tensor (numpy/torch/jax/list) -> flat torch.Tensor
  single_token.py    SingleTokenSafetyProbe (probe_kind="single_shot")
  trajectory.py      TrajectorySafetyProbe (probe_kind="trajectory")
  custom_mlp.py      CustomMLPProbe (probe_kind="single_shot") -- loads a REAL checkpoint from disk
  spec_binding.py    PROBE_REGISTRY, resolve_probe_cls, probe_kwargs_from_extraction_point
examples/
  content_safety_llama.yaml    sample ExtractionPoint spec for a Llama-family model
  content_safety_gpt2.yaml     same spec, retargeted to gpt2's layer count -- used by the vLLM pipeline
  custom_mlp_llama31_8b.yaml   spec for CustomMLPProbe against Llama-3.1-8B
  run_vllm_pipeline.py         reference e2e pipeline: spec -> Router -> live vLLM generation -> probe verdicts
  make_dummy_mlp_checkpoint.py writes a placeholder CustomMLPProbe checkpoint (random weights)
  serve_llama_mlp_pipeline.py  HTTP server: CustomMLPProbe + Llama-3.1-8B, concurrent requests
tests/                        unit tests (pytest) + the live integration stub
```

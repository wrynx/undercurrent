# Changelog

All notable changes to Undercurrent are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While Undercurrent is on 0.x:

- A **minor** release (0.1 → 0.2) may contain breaking changes. Each one is
  listed under **Changed** or **Removed** with what to do about it.
- A **patch** release (0.1.0 → 0.1.1) contains only backwards-compatible fixes.
- A deprecated API keeps working and emits a `DeprecationWarning` for at least
  one minor release before it is removed.

Versions come from git tags (`vX.Y.Z`). See [RELEASING.md](RELEASING.md) for
how a release is cut.

## [Unreleased]

## [0.2.0] - 2026-10-31

Performance and correctness work on the vLLM adapter, and experimental
graph-mode `residual_stream` capture that keeps CUDA graphs and
`torch.compile`. There are no breaking changes: code written for 0.1.0 runs
unchanged.

### Added

- vLLM adapter (experimental): graph-mode `residual_stream` capture.
  `load_model(graph_capture_layers=[...])` (also accepted by
  `ProbedModel(..., backend="vllm")`) captures the residual stream at the given
  decoder layers without `enforce_eager=True`, so the engine keeps CUDA graphs
  and `torch.compile`. It reuses vLLM's EAGLE-3 auxiliary hidden-state outputs,
  enabled by the `vllm.general_plugins` entry point before compilation. Other
  tensor types, other layers, models without that interface, and speculative
  decoding that uses aux outputs are rejected with an error. The benchmark
  harness gains `--graph-capture`.

### Changed

- vLLM adapter: activation capture on CUDA no longer blocks the host once per
  captured row. Each forward hook gathers its matched rows on the GPU and
  starts one non-blocking copy into pinned host memory; records are built after
  one wait per step (at the end of the step in-process, or when the driver
  polls on the cross-process path). Probes see the same records, in the same
  order, with inline aborts taking effect at the same step as before. See
  "Copying captures to the host" in the vLLM adapter internals docs.
- vLLM adapter: on the cross-process polling path (vLLM 0.28's default), the
  worker sends each poll's activations as one packed batch (float32 `bytes`
  plus per-field lists) instead of one dict per record with the tensor as a
  nested Python list, and the driver routes each drain in one executor call
  instead of one per record. In a CPU microbenchmark at a 3,584-wide hidden
  size, transport fell from about 350 µs to under 15 µs per record. Probes see
  the same records in the same order; aborts are sent after the drain is
  routed.
- vLLM adapter: forward hooks skip (layer, tensor type) pairs no request
  captures with one set lookup, and position matching runs once per step
  instead of once per hooked layer; the captured tensor is only extracted when
  something matched. Hook-path CPU time per step fell by about a third in a
  microbenchmark (16 requests x 8 probes, 28 layers).

### Fixed

- vLLM adapter: `ActivationRecord.timestamp` is now the capture time, as
  documented, on every path. On the cross-process polling path the record was
  rebuilt in the driver with a fresh timestamp, so a probe (or a latency
  measurement) saw the time of routing, not of capture, and couldn't tell how
  stale an activation was. The worker now stamps each capture when its hook
  runs and the packed batch carries it to the driver.
- vLLM adapter: with `OverflowPolicy.BLOCK` on the cross-process polling path,
  async probes ran one at a time instead of in parallel. Each drained backlog
  was routed one request at a time, so `route()` waited on one request's full
  queue while every other request's worker sat idle. Drained records are now
  interleaved round-robin across requests, keeping each request's forward-pass
  order. On an A100 (Qwen2.5-7B, 16 requests, 76 ms probe) the old order made
  `block` about 13x slower than the probe's own pace. The docs now also say
  that on this path `block` delays results rather than slowing the engine.
- vLLM adapter: activations could be silently dropped under concurrency on the
  polling path (engine core in a subprocess, the vLLM 0.28 default). One
  request's drain of the shared worker buffer could still be routing another
  request's records when that request received its final output and ended, so
  those records were discarded as late. A request that finishes in the step that
  captured its only extraction point (e.g. `prompt[-1]` with one new token) lost
  its capture entirely; longer generations could lose their last activations.
  Drains are now serialised, so every record captured up to a request's final
  step is routed before the request ends.

## [0.1.0] - 2026-10-02

The first public release of **Undercurrent**, Wrynx's activation-probing
platform for LLM inference. Earlier versions were internal development builds
and were never published.

Planned work that is not part of this release (an OpenAI-compatible
`undercurrent serve` server, Hugging Face Hub integration, a published
container image, and more) is tracked in the
[roadmap](https://github.com/wrynx/undercurrent/blob/main/ROADMAP.md).

### Added

- **One installable package.** `pip install undercurrent` installs the
  `undercurrent` package for Python 3.10+, with the Hugging Face
  `transformers` backend in the base install. vLLM is never installed by
  default. Tested on Python 3.10–3.13 with torch 2.1+ and transformers
  4.40–5.x; see [Compatibility](https://wrynx.github.io/undercurrent/compatibility/).
- **`ProbedModel`**, the high-level API (`undercurrent.ProbedModel`):
  `ProbedModel.from_pretrained(model, spec=..., probes=...)`, then
  `generate()`, which returns a `GenerationOutput` with the text and each
  extraction point's `ProbeResult`. Backends `"hf"` (default) and `"vllm"`.
  Pass `on_result=callback` for a simple result hook alongside the sink API.
- **Extraction-point specs** (`undercurrent.spec`). Declare what to capture
  in YAML (`load_yaml_file`, `parse_yaml`, `parse_dict`) or construct
  `ExtractionPoint`s in Python. Specs are validated by pydantic models and
  round-trip back to YAML (`to_yaml`).
  - Tensor types `residual_stream`, `attn_out`, `mlp_out` and `final_norm`,
    selected per layer.
  - A position-selector grammar: absolute indices (`5`), prompt and generated
    indices (`prompt[-1]`, `generated[0]`), every generated token
    (`generated[*]`), generated slices (`generated[5:]`, `generated[2:8]`),
    and offsets (`prompt[-1]+1`). Malformed selectors raise
    `PositionSyntaxError` with the accepted forms.
  - `until` conditions for continuous selectors (`generation_end`,
    `fixed_count`, `stop_token`).
  - Example specs in `examples/specs/`.
- **Probe interface** (`undercurrent.core`). Probes implement the lifecycle
  `spawn` → `on_start` → `on_activation`* → `on_end` and return
  `ProbeSignal`s and a final `ProbeResult`. There are two probe kinds,
  `single_shot` and `trajectory`. A fresh instance is spawned for each
  request and extraction point, and probe classes that declare mutable
  class-level attributes are rejected when the class is defined, so instances
  can't share state across requests. Reference probes are in
  `undercurrent.core.examples`: `MLPClassifierProbe` (single-shot) and
  `TrajectoryScoreProbe` (trajectory).
- **Function probes and a probe registry.** Decorate a plain function
  `(record) -> float | bool | ProbeSignal | None` with `@probe("name", ...)`
  for stateless single-shot probes, and register probe classes by name with
  `@register_probe("name")`. Third-party packages can expose probes through
  the `undercurrent.probes` entry-point group.
- **Router** (`undercurrent.router`). Dispatches activation records to
  per-request probe instances. Each extraction point picks an execution mode:
  - `inline`: the probe runs on the generation path and its signal is
    returned to the adapter, so it can abort generation.
  - `async`: the probe runs out of band and doesn't block the response. Each
    binding has a bounded queue (`DEFAULT_QUEUE_DEPTH`), served by a shared
    worker pool, with a configurable `OverflowPolicy` (`drop_oldest`,
    `drop_newest` or `block`).
- **Interventions.** An inline probe can return `ProbeAction.ABORT` to stop
  generation. With the `block_until_signal` intervention policy, the router
  waits up to `timeout_ms` for the probe, then applies `on_timeout`
  (`continue` or `abort`). A circuit breaker downgrades an extraction point
  to observe-only after `circuit_breaker_threshold` consecutive timeouts or
  probe errors, so a failing probe can't keep stalling generation.
- **Metrics.** Per-binding queue depth, drops, activation latency and probe
  errors go to a pluggable `MetricsSink`. `InMemoryMetricsRegistry` provides
  snapshots.
- **Observation sinks** (`undercurrent.sinks`) for async probe output:
  - `FileLogSink` writes NDJSON.
  - `WebhookLogSink` POSTs JSON from a background thread, so a slow endpoint
    never adds latency to the caller. Failed deliveries are retried with
    exponential backoff, records that exhaust their retries are
    dead-lettered to a file, and the send queue is bounded.
  - A `LogSink` base class for writing your own sink.
- **Sink redaction.** `redact_keys()`, `drop_keys()` and `chain()` control
  what a sink writes. `WebhookLogSink` redacts prompt text by default.
- **Router context managers.** `Router` and `router.request(...)` can be used
  in `with` blocks so requests end and workers shut down cleanly.
- **Hugging Face adapter** (`undercurrent.adapters.hf`, in the base install).
  `HFEngineAdapter` captures activations from `transformers` models with
  forward hooks during `generate()`, and aborts generation through a
  `StoppingCriteria` (`ProbingStoppingCriteria`).
- **vLLM adapter** (`undercurrent.adapters.vllm`). Install Undercurrent into
  your existing vLLM environment or image (the recommended production path),
  or use the `undercurrent[vllm]` extra for a vLLM from the tested range.
  - Capture is aware of continuous batching: a worker extension installs the
    hooks, and `SeqIdMapper` translates batch rows to
    `(request_id, token_pos)`. Abort signals are wired back to the engine.
  - A `vllm.general_plugins` entry point registers the worker extension.
  - Constructing the adapter checks the installed vLLM version against the
    supported range and raises `VLLMAdapterLimitationError` with the
    installed version and the supported range. Set
    `UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM=1` to downgrade the error to a
    warning. See
    [docs/compatibility.md](https://github.com/wrynx/undercurrent/blob/main/docs/compatibility.md).
  - Tensor and pipeline parallelism: results from every worker are merged,
    and events that tensor-parallel ranks duplicate are deduplicated.
    Multi-worker topologies are refused unless you pass
    `allow_unsupported_executor=True`, because cross-rank ordering under
    pipeline parallelism and Ray/distributed executors haven't been
    validated. See the
    [tensor and pipeline parallelism notes](https://wrynx.github.io/undercurrent/internals/vllm-parallelism/).
- **Command-line tool** `undercurrent`: `inspect-model` lists a model's
  hookable layers and supported tensor types, `validate` checks spec files,
  and `schema` prints the spec JSON Schema (also in
  `schema/probe-spec.schema.json` and `undercurrent.spec.json_schema()`).
- **Curated top-level API.** The common names import from `undercurrent`
  directly; see [API stability](https://wrynx.github.io/undercurrent/api-stability/)
  for what is public. `load_spec()`
  loads a spec from a path, YAML string or dict. The package ships `py.typed`.
- **Error hierarchy.** Errors Undercurrent raises on purpose derive from
  `ProbingError` (and the matching built-in type), with messages that say
  how to fix the problem.
- **Clear errors for missing backends.** Using an adapter whose backend isn't
  installed raises `MissingDependencyError`, which says how to install it.
- **Examples** (in the repository, not installed with the package, and covered
  by the test suite):
  - [`examples/content_safety/`](https://github.com/wrynx/undercurrent/tree/main/examples/content_safety):
    a **demo** content-safety probe in both `single_shot` and `trajectory`
    variants, with YAML specs, a dummy-checkpoint maker, a vLLM pipeline
    script and a Dockerfile. It is a demonstration of the probe API, not a
    trained safety classifier.
  - [`examples/openai_server/`](https://github.com/wrynx/undercurrent/tree/main/examples/openai_server):
    a reference OpenAI-compatible wire format for probe verdicts (completion
    bodies and SSE chunks) and a reference HTTP server built on it.
- **Probe-training example** (`examples/train_probe/`) and Colab notebooks
  (`notebooks/quickstart.ipynb`, `notebooks/train_probe.ipynb`).

### Fixed

- **vLLM `residual_stream` on fused-residual layers.** On Llama-family models
  (and the other vLLM architectures whose decoder layer returns
  `(hidden_states, residual)`), a `residual_stream` extraction point captured
  the layer's MLP output. It now captures `hidden_states + residual`, the
  residual stream after the layer, matching the HF backend. Decoder-layer
  outputs the adapter can't classify raise `VLLMAdapterLimitationError`
  instead of being captured silently. Verified on GPU (NVIDIA L4, vLLM 0.28.0)
  by `tests/adapters/vllm/test_residual_stream_gpu.py`, which compares the
  vLLM and HF captures token by token.

### Known issues

See [Known issues (v0.1)](https://github.com/wrynx/undercurrent/blob/main/docs/compatibility.md#known-issues-v01)
for details and workarounds.

- **No GPU CI.** The GPU tests run by hand with `scripts/gpu_check.sh` before
  each release. 0.1.0 passed on an NVIDIA L4 with vLLM 0.28.0, torch 2.13.0
  and CUDA 13.0; other GPUs and multi-GPU topologies haven't been run.
- **Only vLLM 0.28 is supported** (`vllm>=0.28,<0.29`), and vLLM 0.30 is
  already out. `UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM=1` lets you try a newer
  vLLM; the range widens only after a GPU validation run.

### Security

- Vulnerabilities can be reported privately to security@wrynx.com or through
  GitHub private vulnerability reporting. See
  [SECURITY.md](https://github.com/wrynx/undercurrent/blob/main/SECURITY.md).
- The content-safety example loads classifier checkpoints only with
  `torch.load(..., weights_only=True)`, accepts only plain tensor
  `state_dict`s, and refuses to load on torch older than 2.6 (the first
  release with a fix for CVE-2025-32434).

[Unreleased]: https://github.com/wrynx/undercurrent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/wrynx/undercurrent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/wrynx/undercurrent/releases/tag/v0.1.0

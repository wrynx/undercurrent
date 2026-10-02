# API stability

This page says which parts of Undercurrent are public, what we promise about
them, and how they change between releases.

## What is public

A name is public if it is listed in the `__all__` of one of these modules:

| Module | What it holds |
|--------|---------------|
| `undercurrent` | The front door and the advanced API (see [the two tiers](#the-two-tiers)) |
| `undercurrent.model` | `ProbedModel`, `GenerationOutput`, `ProbedModelConfigError`, defaults |
| `undercurrent.spec` | Extraction-point specs: parsing, resolved types, enums, serialization, JSON Schema |
| `undercurrent.core` | The `Probe` interface, function probes, the probe registry, signal and result types |
| `undercurrent.core.examples` | Reference probes (stub scoring, for tests and demos) |
| `undercurrent.router` | `Router`, async-execution settings, metrics |
| `undercurrent.sinks` | Observation sinks and redaction |
| `undercurrent.adapters` | The `EngineAdapter` ABC and `MissingDependencyError` |
| `undercurrent.adapters.hf` | `HFEngineAdapter` |
| `undercurrent.adapters.vllm` | `VLLMEngineAdapter` (see [experimental](#experimental-parts)) |
| `undercurrent.errors` | `ProbingError` and its helper subclasses |
| `undercurrent.cli` | `main`, the console-script entry point |

The `undercurrent` command line (its commands, options and exit codes) and the
YAML spec format (see `undercurrent schema`) are public too.

Everything else is internal, even if you can import it:

- modules and names that start with an underscore;
- names that a module imports but doesn't list in its `__all__`. For example,
  `undercurrent.router` still lets you import `ProbeFactory`, but that name
  belongs to `undercurrent.core`, so import it from there (or from
  `undercurrent`);
- modules whose docstring says they are internal, such as
  `undercurrent.router.binding`, `undercurrent.sinks.records`,
  `undercurrent.adapters.hf.stopping_criteria` and the `undercurrent.cli`
  subcommand modules.

Internal code can change in any release without notice. If you need something
internal, please [open an issue](https://github.com/wrynx/undercurrent/issues)
so we can consider making it public.

A test (`tests/test_public_api.py`) pins every public module's `__all__`, so
the public surface can't change by accident.

## The two tiers

### Front door

The names most people need, listed first in `undercurrent.__all__`:

```python
from undercurrent import (
    # load a model, generate, read results
    ProbedModel,
    GenerationOutput,
    # write and register probes
    probe,
    register_probe,
    Probe,
    # say what to capture
    ExtractionPoint,
    load_spec,
    # catch any error Undercurrent raises on purpose
    ProbingError,
    __version__,
)
```

We try hardest to keep these stable. Changes to them are rare, and we give
them a deprecation period even while Undercurrent is on 0.x (see below).

### Advanced

The API for embedding Undercurrent in your own serving stack. It is also
exported from `undercurrent`:

- the `Router` and `RequestHandle`;
- probe data types: `ProbeFactory`, `ProbeSignal`, `ProbeAction`,
  `ProbeResult`, `RequestContext`, `ActivationRecord`, `ProbeSpec`;
- spec enums and helpers: `TensorType`, `ProbeKind`, `ExecutionMode`,
  `parse_position`;
- policies you configure: `OverflowPolicy`, `InterventionPolicy`,
  `InterventionMode`, `TimeoutAction`;
- observation sinks: `LogSink`, `FileLogSink`, `WebhookLogSink`, and the
  redaction helpers `redact_keys`, `drop_keys`, `chain`;
- metrics: `MetricsSink`, `InMemoryMetricsRegistry`, `MetricsSnapshot`;
- `EngineAdapter`, for writing an adapter for another engine;
- errors worth catching by name: `SpecValidationError`, `ProbeNotFoundError`,
  `RouterError`, `MissingDependencyError`.

The subpackages (`undercurrent.spec`, `undercurrent.core`, ...) list a few more
advanced names in their own `__all__`.

The advanced API follows the same versioning rules as the front door. It is
larger and closer to the internals, so expect it to change more often while
Undercurrent is on 0.x. Every change is listed in the
[changelog](https://github.com/wrynx/undercurrent/blob/main/CHANGELOG.md).

The concrete adapters, `HFEngineAdapter` and `VLLMEngineAdapter`, aren't
exported from `undercurrent`. Import them from `undercurrent.adapters.hf` and
`undercurrent.adapters.vllm`. Most users never need them, because
`ProbedModel` picks one for you with `backend="hf"` or `backend="vllm"`.

## Versioning

Undercurrent follows [Semantic Versioning](https://semver.org/). While it is
on `0.x`:

- a **minor** release (`0.1` → `0.2`) may break the public API. Each break is
  listed in the changelog under **Changed** or **Removed**, with what to do
  about it;
- a **patch** release (`0.1.0` → `0.1.1`) never breaks it. Patch releases
  contain only backwards-compatible fixes.

From `1.0` on, only a major release may break the public API.

## Deprecation policy

Before a public name is removed or changes incompatibly:

1. it keeps working, and emits a `DeprecationWarning` that says what to use
   instead, for **at least one minor release**;
2. the deprecation is noted in the changelog under **Deprecated** in the
   release that introduces it, and the removal under **Removed** in the
   release that completes it.

For example, the spec keys `layer` and `tensor` still work as aliases for
`layers` and `tensor_type`, and emit a `DeprecationWarning`.

To see deprecation warnings in your own tests, run them with
`python -W error::DeprecationWarning` (or pytest's `-W` option).

## Experimental parts

Some parts depend on things outside our control and are **experimental**: they
can change in any release, including a patch release, when the engine they
depend on changes.

- **vLLM adapter internals.** `undercurrent.adapters.vllm` reads undocumented
  vLLM internals (the V1 model-runner batch layout and the
  `worker_extension_cls` mechanism). `VLLMEngineAdapter` and
  `ProbedModel(..., backend="vllm")` are public. The modules behind them
  (`seq_mapper`, `worker_extension`, `introspection`, `plugin`, `support`,
  `version_check`) are internal and experimental. Each Undercurrent release
  supports a specific vLLM range and checks it at runtime.
- **Captured tensor layouts.** Which tensor a `tensor_type` maps to depends on
  the model architecture and the engine. We document the mapping, but a new
  model family or engine version can change it.

## Supported Python and engine versions

See [Compatibility](compatibility.md) for the supported Python, torch,
transformers and vLLM versions, the combinations we have tested, and the
support policy for engine versions.

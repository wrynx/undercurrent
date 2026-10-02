# Overview

Every public name in Undercurrent, with a one-line summary and a link to its
full reference. Most of them are also exported from the top-level
`undercurrent` package:

```python
from undercurrent import ProbedModel, probe, register_probe
```

!!! tip "Start with `ProbedModel`"
    [`ProbedModel`][undercurrent.model.ProbedModel] is the front door: load a
    model with `ProbedModel.from_pretrained(...)`, attach probes with a spec,
    and call `generate(...)`. Write probes with
    [`@probe`][undercurrent.core.function_probe.probe] or a
    [`Probe`][undercurrent.core.Probe] subclass. The
    [Quickstart](../getting-started/quickstart.md) walks through it.

    The **router**, **sinks**, **metrics** and **adapters** are the
    **advanced** API, for embedding Undercurrent in your own serving stack.

## API stability

A name is public if it is listed in the `__all__` of the top-level package
or of one of the subpackages below; everything else is internal, even if
you can import it. The front door is the most stable part; the advanced
API follows the same versioning rules but may change more often while
Undercurrent is on 0.x. The vLLM adapter's internals are experimental. See
[API stability](../api-stability.md) for the full policy, including SemVer
and deprecations.

## Front door

Exported from `undercurrent`.

| Name | Summary | Page |
|------|---------|------|
| [`ProbedModel`][undercurrent.model.ProbedModel] | **Start here.** A model with probes attached: `from_pretrained(...)`, then `generate(...)`. | [`undercurrent.model`](model.md) |
| [`GenerationOutput`][undercurrent.model.GenerationOutput] | The result of one `ProbedModel.generate()` call for one prompt. | [`undercurrent.model`](model.md) |
| [`probe`][undercurrent.core.function_probe.probe] | Decorator: turn a plain function `fn(record) -> score` into a single-shot probe. | [`undercurrent.core`](core.md) |
| [`register_probe`][undercurrent.core.register_probe] | Register a probe under a ``probe_type`` name in the global registry. | [`undercurrent.core`](core.md) |
| [`Probe`][undercurrent.core.Probe] | Base class for all probes. | [`undercurrent.core`](core.md) |
| [`ExtractionPoint`][undercurrent.spec.ExtractionPoint] | What to capture, where, and which probe gets it. | [`undercurrent.spec`](spec.md) |
| [`load_spec`][undercurrent.spec.load_spec] | Load a spec from whatever form you have it in. | [`undercurrent.spec`](spec.md) |
| [`ProbingError`][undercurrent.errors.ProbingError] | Base class for every exception Undercurrent raises deliberately. | [Errors](#errors) |
| [`__version__`](#version) | The installed version of Undercurrent, as a string. | [Version](#version) |

## Advanced API

Also exported from `undercurrent`: the router, probe data types, spec enums,
sinks, metrics and the adapter interface.

| Name | Summary | Page |
|------|---------|------|
| [`ActivationRecord`][undercurrent.spec.ActivationRecord] | One captured activation, tied to a single extraction point and token. | [`undercurrent.spec`](spec.md) |
| [`chain`][undercurrent.sinks.chain] | Compose redaction functions left to right. If any returns ``None``, the record is dropped. | [`undercurrent.sinks`](sinks.md) |
| [`drop_keys`][undercurrent.sinks.drop_keys] | Remove every matching key (and its value) from the record. | [`undercurrent.sinks`](sinks.md) |
| [`EngineAdapter`][undercurrent.adapters.EngineAdapter] | Abstract base every engine adapter implements. | [`undercurrent.adapters`](adapters.md) |
| [`ExecutionMode`][undercurrent.spec.ExecutionMode] | Inline (on the generation path, can abort) or async (worker pool, observe only). | [`undercurrent.spec`](spec.md) |
| [`FileLogSink`][undercurrent.sinks.FileLogSink] | Appends one NDJSON line per signal/result to ``path``. | [`undercurrent.sinks`](sinks.md) |
| [`InMemoryMetricsRegistry`][undercurrent.router.InMemoryMetricsRegistry] | Thread-safe in-memory counters and gauges, keyed by (request_id, extraction_point_name). | [`undercurrent.router`](router.md) |
| [`InterventionMode`][undercurrent.spec.InterventionMode] | How an inline extraction point's ``ProbeSignal`` relates to the engine's decode step. | [`undercurrent.spec`](spec.md) |
| [`InterventionPolicy`][undercurrent.spec.InterventionPolicy] | How an inline extraction point's probe may hold up generation. | [`undercurrent.spec`](spec.md) |
| [`LogSink`][undercurrent.sinks.LogSink] | Out-of-band destination for a probe's intermediate signals and final verdict. | [`undercurrent.sinks`](sinks.md) |
| [`MetricsSink`][undercurrent.router.MetricsSink] | Pluggable destination for async-binding metrics events. | [`undercurrent.router`](router.md) |
| [`MetricsSnapshot`][undercurrent.router.MetricsSnapshot] | Point-in-time read of one async binding's metrics. | [`undercurrent.router`](router.md) |
| [`MissingDependencyError`][undercurrent.adapters.MissingDependencyError] | Raised when an optional package Undercurrent needs isn't installed (most often vLLM). | [`undercurrent.adapters`](adapters.md) |
| [`OverflowPolicy`][undercurrent.router.OverflowPolicy] | What an async extraction point's bounded queue does when it is full. | [`undercurrent.router`](router.md) |
| [`parse_position`][undercurrent.spec.parse_position] | Parse a raw YAML ``position`` value into a `PositionSelector`. | [`undercurrent.spec`](spec.md) |
| [`ProbeAction`][undercurrent.core.ProbeAction] | What a signal asks for: continue, abort or flag. | [`undercurrent.core`](core.md) |
| [`ProbeFactory`][undercurrent.core.ProbeFactory] | Binds a Probe subclass to fixed spawn kwargs. | [`undercurrent.core`](core.md) |
| [`ProbeKind`][undercurrent.spec.ProbeKind] | A probe's shape: single-shot or trajectory. | [`undercurrent.spec`](spec.md) |
| [`ProbeNotFoundError`][undercurrent.core.ProbeNotFoundError] | Raised when no probe is registered under a requested ``probe_type``. | [`undercurrent.core`](core.md) |
| [`ProbeResult`][undercurrent.core.ProbeResult] | Returned by `Probe.on_end`, exactly once per (request, extraction point). | [`undercurrent.core`](core.md) |
| [`ProbeSignal`][undercurrent.core.ProbeSignal] | Returned by ``Probe.on_activation`` when an activation warrants a message. | [`undercurrent.core`](core.md) |
| [`ProbeSpec`][undercurrent.spec.ProbeSpec] | A fully resolved spec: an ordered collection of extraction points. | [`undercurrent.spec`](spec.md) |
| [`redact_keys`][undercurrent.sinks.redact_keys] | Replace the value of every matching key with `replacement`. | [`undercurrent.sinks`](sinks.md) |
| [`RequestContext`][undercurrent.core.RequestContext] | What a probe knows about its request, passed to ``on_start`` and ``on_end``. | [`undercurrent.core`](core.md) |
| [`RequestHandle`][undercurrent.router.RequestHandle] | One request's lifecycle on a `Router`, returned by `Router.request(...)`. | [`undercurrent.router`](router.md) |
| [`Router`][undercurrent.router.Router] | Dispatches each `ActivationRecord` to the probe instances of its request. | [`undercurrent.router`](router.md) |
| [`RouterError`][undercurrent.router.RouterError] | Raised for invalid Router usage. | [`undercurrent.router`](router.md) |
| [`SpecValidationError`][undercurrent.spec.SpecValidationError] | Raised when a spec fails validation. | [`undercurrent.spec`](spec.md) |
| [`TensorType`][undercurrent.spec.TensorType] | Which tensor an extraction point captures at each of its layers (``tensor_type`` in a spec). | [`undercurrent.spec`](spec.md) |
| [`TimeoutAction`][undercurrent.spec.TimeoutAction] | What a `block_until_signal` wait falls back to on timeout: continue or abort. | [`undercurrent.spec`](spec.md) |
| [`WebhookLogSink`][undercurrent.sinks.WebhookLogSink] | POSTs each signal/result as JSON to ``url`` from a background thread. | [`undercurrent.sinks`](sinks.md) |

## More in the subpackages

These public names are imported from their subpackage, not from
`undercurrent`.

### `undercurrent.model`

| Name | Summary | Page |
|------|---------|------|
| [`DEFAULT_MAX_NEW_TOKENS`][undercurrent.model.DEFAULT_MAX_NEW_TOKENS] | Generation length used when none is given. | [`undercurrent.model`](model.md) |
| [`DEFAULT_VLLM_MAX_CONCURRENCY`][undercurrent.model.DEFAULT_VLLM_MAX_CONCURRENCY] | How many `generate()` calls the vLLM backend runs at once by default. | [`undercurrent.model`](model.md) |
| [`ProbedModelConfigError`][undercurrent.model.ProbedModelConfigError] | The spec, probes or backend given to `ProbedModel` don't fit together. | [`undercurrent.model`](model.md) |

### `undercurrent.spec`

| Name | Summary | Page |
|------|---------|------|
| [`extraction_point_to_dict`][undercurrent.spec.extraction_point_to_dict] | Convert one resolved extraction point back to a plain (spec-shaped) dict. | [`undercurrent.spec`](spec.md) |
| [`FrozenArgs`][undercurrent.spec.FrozenArgs] | Read-only, picklable mapping used for `ExtractionPoint.probe_args`. | [`undercurrent.spec`](spec.md) |
| [`json_schema`][undercurrent.spec.json_schema.json_schema] | Return the JSON Schema (Draft 2020-12) for probe-spec YAML files. | [`undercurrent.spec`](spec.md) |
| [`load_yaml_file`][undercurrent.spec.load_yaml_file] | Load, parse, and resolve a spec from a YAML file on disk. | [`undercurrent.spec`](spec.md) |
| [`parse_dict`][undercurrent.spec.parse_dict] | Validate a plain dict (already loaded from YAML/JSON) and resolve it. | [`undercurrent.spec`](spec.md) |
| [`parse_yaml`][undercurrent.spec.parse_yaml] | Parse and resolve a spec from a YAML (or JSON, which is valid YAML) string. | [`undercurrent.spec`](spec.md) |
| [`PositionKind`][undercurrent.spec.PositionKind] | The form of a ``position`` selector, as parsed into a `PositionSelector`. | [`undercurrent.spec`](spec.md) |
| [`PositionSelector`][undercurrent.spec.PositionSelector] | Normalized, adapter-facing representation of a ``position`` selector. | [`undercurrent.spec`](spec.md) |
| [`PositionSyntaxError`][undercurrent.spec.PositionSyntaxError] | Raised when a ``position`` string does not match any supported form. | [`undercurrent.spec`](spec.md) |
| [`probe_spec_to_dict`][undercurrent.spec.probe_spec_to_dict] | Convert a resolved ProbeSpec back to a plain (spec-shaped) dict. | [`undercurrent.spec`](spec.md) |
| [`ProbingSpecError`][undercurrent.spec.ProbingSpecError] | Base class for all errors raised by `undercurrent.spec`. | [`undercurrent.spec`](spec.md) |
| [`to_yaml`][undercurrent.spec.to_yaml] | Serialize a resolved ProbeSpec to a YAML string. | [`undercurrent.spec`](spec.md) |
| [`UntilKind`][undercurrent.spec.UntilKind] | When a continuous extraction point stops capturing (``until`` in a spec). | [`undercurrent.spec`](spec.md) |

### `undercurrent.core`

| Name | Summary | Page |
|------|---------|------|
| [`default_registry`][undercurrent.core.default_registry] | The process-wide registry behind the module-level functions and ``Router()``. | [`undercurrent.core`](core.md) |
| [`FunctionProbe`][undercurrent.core.FunctionProbe] | Base class of every class `@probe` creates. Not used directly. | [`undercurrent.core`](core.md) |
| [`get_probe_factory`][undercurrent.core.get_probe_factory] | Look ``name`` up in `default_registry`. | [`undercurrent.core`](core.md) |
| [`list_probes`][undercurrent.core.list_probes] | Sorted names of every probe in `default_registry`, including plugins. | [`undercurrent.core`](core.md) |
| [`ProbeRegistry`][undercurrent.core.ProbeRegistry] | A thread-safe mapping of ``probe_type`` names to `ProbeFactory`. | [`undercurrent.core`](core.md) |
| [`unregister_probe`][undercurrent.core.unregister_probe] | Remove ``name`` from `default_registry`. | [`undercurrent.core`](core.md) |

### `undercurrent.core.examples`

| Name | Summary | Page |
|------|---------|------|
| [`MLPClassifierProbe`][undercurrent.core.examples.MLPClassifierProbe] | single_shot probe: one activation in, one classification verdict out. | [`undercurrent.core`](core.md) |
| [`TrajectoryScoreProbe`][undercurrent.core.examples.TrajectoryScoreProbe] | trajectory probe: running-mean score with abort-on-threshold. | [`undercurrent.core`](core.md) |

### `undercurrent.router`

| Name | Summary | Page |
|------|---------|------|
| [`DEFAULT_CIRCUIT_BREAKER_THRESHOLD`][undercurrent.router.DEFAULT_CIRCUIT_BREAKER_THRESHOLD] | Default ``Router(circuit_breaker_threshold=...)``. | [`undercurrent.router`](router.md) |
| [`DEFAULT_DRAIN_TIMEOUT`][undercurrent.router.DEFAULT_DRAIN_TIMEOUT] | Default ``Router(drain_timeout=...)``, in seconds. | [`undercurrent.router`](router.md) |
| [`DEFAULT_QUEUE_DEPTH`][undercurrent.router.DEFAULT_QUEUE_DEPTH] | Default ``Router(default_queue_depth=...)``. | [`undercurrent.router`](router.md) |
| [`default_worker_pool_size`][undercurrent.router.default_worker_pool_size] | The default ``worker_pool_size``: ``min(32, os.cpu_count() * 4)``. | [`undercurrent.router`](router.md) |
| [`RequestEndListener`][undercurrent.router.RequestEndListener] | ``listener(request_id, results)``, registered with `Router.on_request_end`. | [`undercurrent.router`](router.md) |

### `undercurrent.sinks`

| Name | Summary | Page |
|------|---------|------|
| [`DEFAULT_PROMPT_TEXT_KEYS`][undercurrent.sinks.DEFAULT_PROMPT_TEXT_KEYS] | Keys that can carry prompt or generated text, redacted by ``WebhookLogSink`` by default. | [`undercurrent.sinks`](sinks.md) |
| [`RedactFn`][undercurrent.sinks.RedactFn] | A redaction hook: takes a JSON-able record, returns it (possibly changed) or ``None`` to drop it. | [`undercurrent.sinks`](sinks.md) |
| [`to_jsonable`][undercurrent.sinks.to_jsonable] | Recursively convert `value` into something `json.dumps` can handle. | [`undercurrent.sinks`](sinks.md) |
| [`wire_router`][undercurrent.sinks.wire_router] | Attach ``sink`` to ``router``; the same as ``router.attach_log_sink(sink)``. | [`undercurrent.sinks`](sinks.md) |

### `undercurrent.adapters.hf`

| Name | Summary | Page |
|------|---------|------|
| [`HFAdapterLimitationError`][undercurrent.adapters.hf.HFAdapterLimitationError] | Raised for a documented, known limitation of the HF adapter. | [`undercurrent.adapters`](adapters.md) |
| [`HFEngineAdapter`][undercurrent.adapters.hf.HFEngineAdapter] | The `EngineAdapter` for Hugging Face transformers models. | [`undercurrent.adapters`](adapters.md) |

### `undercurrent.adapters.vllm`

| Name | Summary | Page |
|------|---------|------|
| [`VLLMAdapterLimitationError`][undercurrent.adapters.vllm.VLLMAdapterLimitationError] | Raised for a documented, known limitation of the vLLM adapter. | [`undercurrent.adapters`](adapters.md) |
| [`VLLMEngineAdapter`][undercurrent.adapters.vllm.VLLMEngineAdapter] | The `EngineAdapter` for vLLM (V1 engine), with concurrent ``generate()`` calls. | [`undercurrent.adapters`](adapters.md) |

### `undercurrent.errors`

| Name | Summary | Page |
|------|---------|------|
| [`ProbeDefinitionError`][undercurrent.errors.ProbeDefinitionError] | A probe class or ``@probe`` function is defined in a way Undercurrent can't use. | [Errors](#errors) |
| [`ProbingKeyError`][undercurrent.errors.ProbingKeyError] | A name lookup missed (a ``KeyError``), with a readable message. | [Errors](#errors) |
| [`ProbingRuntimeError`][undercurrent.errors.ProbingRuntimeError] | A library object was used in a state that doesn't allow it (a ``RuntimeError``). | [Errors](#errors) |
| [`ProbingTypeError`][undercurrent.errors.ProbingTypeError] | A library call got an object of the wrong type (a ``TypeError``). | [Errors](#errors) |
| [`ProbingValueError`][undercurrent.errors.ProbingValueError] | A library call got an invalid value (a ``ValueError``). | [Errors](#errors) |
| [`SpecFileNotFoundError`][undercurrent.errors.SpecFileNotFoundError] | A spec file path doesn't exist (a ``FileNotFoundError``). | [Errors](#errors) |

### `undercurrent.cli`

| Name | Summary | Page |
|------|---------|------|
| [`main`][undercurrent.cli.main] | Run the ``undercurrent`` CLI and return its exit code. | [CLI](cli.md) |

## Errors

Every exception Undercurrent raises on purpose is a
[`ProbingError`][undercurrent.errors.ProbingError]. Each one also keeps a
built-in base (`SpecValidationError` is a `ValueError`, `ProbeNotFoundError`
a `KeyError`, `MissingDependencyError` an `ImportError`, ...), so existing
`except` clauses keep working. Errors raised by your own code (a probe's
`on_activation`, an `on_result` callback) pass through unchanged.

```python
from undercurrent import ProbedModel, ProbingError

try:
    model = ProbedModel.from_pretrained("gpt2", spec="probes.yaml")
except ProbingError as exc:
    print(f"configuration problem: {exc}")
```

The more specific errors are documented with the module that raises them:
[`SpecValidationError`][undercurrent.spec.SpecValidationError],
[`ProbeNotFoundError`][undercurrent.core.ProbeNotFoundError],
[`RouterError`][undercurrent.router.RouterError],
[`MissingDependencyError`][undercurrent.adapters.MissingDependencyError] and
[`ProbedModelConfigError`][undercurrent.model.ProbedModelConfigError].

::: undercurrent.errors.ProbingError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.errors.ProbingValueError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.errors.ProbingTypeError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.errors.ProbingRuntimeError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.errors.ProbingKeyError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.errors.ProbeDefinitionError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.errors.SpecFileNotFoundError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Version

`undercurrent.__version__` is the installed version, as a string (for
example `"0.1.0"`). The `undercurrent --version` command prints the same
value.

```python
import undercurrent

print(undercurrent.__version__)
```

# probing_spec

The extraction-point **contract layer** for the activation-probing platform.

This package defines *what* to capture during LLM inference and *how* that
gets expressed as data — nothing more. It has no dependency on any specific
inference engine (no torch/transformers/vllm import anywhere in this
package), and it implements no adapters, routers, or probes. Those
components depend on this one; this one depends on nothing project-specific.

```
YAML spec  --(parse)-->  validated schema  --(resolve)-->  ProbeSpec (ExtractionPoint*)
                                                                    |
                                                    adapters call .matches(...) per token
                                                                    |
                                                                    v
                                                          ActivationRecord (runtime data)
```

## Install

```bash
pip install -e .   # from the repo root; installs undercurrent
```

Requires Python 3.9+. Dependencies: `pydantic>=2`, `PyYAML>=6`.

## Quick start

```python
from undercurrent.spec import load_yaml_file

spec = load_yaml_file("probes.yaml")

for point in spec:
    print(point.name, point.tensor_type, point.layers)

# Adapters call .matches() per token during inference -- the position
# grammar has already been resolved, no re-parsing needed.
point = spec.get("last_prompt_token_probe")
point.matches(token_index=41, is_generated=False, prompt_len=42)  # -> True
```

Invalid specs raise `undercurrent.spec.SpecValidationError` with a message
describing exactly what's wrong — never a warning, and never a raw
`pydantic.ValidationError` or `yaml.YAMLError` leaking out.

## Extraction point fields

| Field             | Type                                    | Required | Notes |
|-------------------|------------------------------------------|----------|-------|
| `name`            | `str`                                    | yes      | unique within the spec file |
| `layers`          | `int` or `list[int]`                     | yes      | one `ActivationRecord` is produced per matched layer; always resolves to a tuple. The old key `layer` is a deprecated alias |
| `tensor_type`     | `residual_stream \| attn_out \| mlp_out \| kv \| final_norm` | yes | the old key `tensor` is a deprecated alias |
| `position`        | see [Position selectors](#position-selectors) | yes | |
| `stride` (`every_n`) | `int`                                  | no       | subsample a continuous `position`; only valid when `position` is continuous |
| `until`           | `generation_end \| fixed_count \| stop_token` | no  | only valid when `position` is continuous |
| `probe_type`      | `str`                                    | yes      | name of a registered probe class (probe registry lives outside this package) |
| `probe_kind`      | `single_shot \| trajectory`              | yes      | |
| `execution_mode`  | `inline \| async`, default `inline`      | no       | `async` requires `probe_kind=trajectory` (see below) |
| `queue_depth`     | `int`                                    | no       | only valid when `execution_mode=async` |
| `intervention`    | see [Intervention policy](#intervention-policy) | no | if omitted, the router applies its own `default_intervention_policy` |
| `probe_args`      | mapping                                  | no       | extra constructor kwargs for the probe, merged over the router's `ProbeFactory.probe_kwargs` (the point wins) |

The YAML keys and the attributes of the resolved `ExtractionPoint` share one
vocabulary (`point.layers`, `point.tensor_type`, `point.probe_args`). The
older keys `layer` and `tensor` are still accepted but emit a
`DeprecationWarning`; setting both an old and a new key for the same field
is a validation error.

### Rejected combinations

Parsing fails fast (raises `SpecValidationError`) for:

- `execution_mode=async` with `probe_kind=single_shot` — async execution
  implies a probe that carries state across calls; a single-shot probe has
  no state to carry, so this combination has no defined semantics.
- `queue_depth` set while `execution_mode=inline`.
- `stride` or `until` set on a non-continuous `position` (an absolute
  index, `prompt[i]`, or `generated[i]` only ever matches one token, so
  neither field means anything there).
- duplicate `name`s within one spec file.
- unknown fields (the schema uses `extra="forbid"`).
- `execution_mode=async` with an `intervention` policy other than the
  default (`mode=reject`) — async is already fire-and-forget (see
  `probing_router`'s `route()`), so it cannot participate in synchronous
  intervention.
- `intervention.timeout_ms` set while `mode=reject` (or missing while
  `mode=block_until_signal`, or `<1`), and `intervention.on_timeout` set to
  anything but `continue` while `mode=reject`.

## Intervention policy

`intervention` (optional, per extraction point) or a router-level default
(`Router(..., default_intervention_policy=...)`, in `probing_router`)
controls whether — and how long — an inline extraction point's decode step
may wait on `on_activation`'s return value before proceeding:

| Field | Type | Notes |
|-------|------|-------|
| `mode` | `reject \| block_until_signal`, default `reject` | `reject`: call `on_activation` and return whatever it produces, no wait-for-decision contract. `block_until_signal`: wait up to `timeout_ms` for it. |
| `timeout_ms` | `int` | **required** when `mode=block_until_signal` (a probe must never be able to hang generation indefinitely); forbidden otherwise. |
| `on_timeout` | `continue \| abort`, default `continue` | fallback action substituted if `timeout_ms` elapses (or the probe raises) with no signal; only meaningful under `block_until_signal`. |

```yaml
intervention:
  mode: block_until_signal
  timeout_ms: 250
  on_timeout: abort
```

The resolved counterpart is `undercurrent.spec.InterventionPolicy` (a frozen
dataclass, same three fields) — `ExtractionPoint.intervention` is `None`
when not specified in the spec at all (meaning "use the router's default"),
never a policy object with implicit values. **Enforcement of `timeout_ms`
itself, the `on_timeout` fallback, and the async rejection above all live
in `probing_router`** (see that package's README/`router.py`) — this
package only validates the *shape* of the policy; it has no dispatch logic
of its own.

## Position selectors

A `position` is either a bare integer or one of these string forms:

| Form                 | Meaning                                                  |
|----------------------|-----------------------------------------------------------|
| `5`                  | absolute index into the full (prompt + generated) sequence |
| `"prompt[i]"`        | the token at index `i` within the prompt (0-based)        |
| `"prompt[-1]"`       | the last prompt token (negative indices count from the end) |
| `"generated[i]"`     | the `i`-th generated token (0-based)                       |
| `"generated[-1]"`    | the last generated token                                   |
| `"generated[*]"`     | every generated token — continuous                          |
| `"generated[5:]"`    | every generated token from index 5 onward — continuous      |
| `"generated[2:8]"`   | generated tokens in `[2, 8)` — continuous                   |
| any of the above `+N` / `-N` | an offset added after resolving the base index, e.g. `"prompt[-1]+1"` = one token past the last prompt token |

Parsing resolves each of these into a `PositionSelector` — a plain, frozen
value type with one method adapters need:

```python
selector.matches(token_index, is_generated, prompt_len, generated_index=None, num_generated_total=None) -> bool
```

`generated_index` and `num_generated_total` are optional and only needed for
continuous selectors (`generated[*]`, slices) and for selectors with a
negative generated index (`generated[-1]`), which can't resolve to an
absolute position until the total generated length is known — during
streaming decode, re-check those once generation ends if you need them.

Adapters should treat `PositionSelector` as opaque and call `.matches(...)`;
they should never parse `position` strings themselves.

## `ActivationRecord`

The one piece of *runtime* data this package defines — what an adapter hands
a probe once an extraction point matches a token:

```python
from undercurrent.spec import ActivationRecord

record = ActivationRecord(
    request_id="req-123",
    extraction_point_name="last_prompt_token_probe",
    layer=12,
    token_pos=41,
    tensor_type="residual_stream",
    tensor=activation,   # numpy array, torch.Tensor, jax array -- anything array-like
    is_generated=False,
)
```

`tensor` is intentionally untyped (`Any`): this package never imports numpy
or torch, so it never forces a framework choice on adapters. Use
`record.metadata()` to get every field except `tensor` as a plain dict
(handy for logging, since tensors usually aren't JSON-serializable).

## Example specs

See [`examples/`](../../examples/specs/) for complete, parseable YAML files:

- [`classic_single_token.yaml`](../../examples/specs/classic_single_token.yaml) —
  single-token probe at the last prompt token, inline, single-shot.
- [`continuous_trajectory.yaml`](../../examples/specs/continuous_trajectory.yaml) —
  continuous trajectory probe over every 2nd generated token, async, across
  three layers at once.
- [`offset_dual_position.yaml`](../../examples/specs/offset_dual_position.yaml) —
  offset-based dual-position probe (`"prompt[-1]+1"`), for "read position b
  at token b+1" style specs.
- [`sliced_range.yaml`](../../examples/specs/sliced_range.yaml) — a fixed slice of
  generated tokens (`"generated[5:20]"`).

## Round-tripping

A resolved `ProbeSpec` can be serialized back to spec-shaped data:

```python
from undercurrent.spec import parse_dict, probe_spec_to_dict, to_yaml

spec = parse_dict(original_dict)
probe_spec_to_dict(spec)   # -> plain dict, spec-shaped
to_yaml(spec)              # -> YAML string
```

`parse_dict(probe_spec_to_dict(spec)) == spec` for any valid spec (see
`tests/spec/test_roundtrip.py`).

## Package layout

```
src/undercurrent/spec/
  errors.py             SpecValidationError, PositionSyntaxError
  position.py           position grammar parser + PositionSelector
  schema.py             pydantic models for the as-written spec (surface syntax + combination rules)
  resolved.py           ExtractionPoint / ProbeSpec -- the resolved contract adapters depend on
  parser.py             YAML/dict -> validated schema -> resolved ProbeSpec
  serialize.py          resolved ProbeSpec -> dict / YAML
  activation_record.py  ActivationRecord
tests/spec/             unit tests (pytest, at the repo root)
examples/specs/         example YAML specs (at the repo root)
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/spec
```

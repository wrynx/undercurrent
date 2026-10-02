# Extraction points & specs

<!-- owner: p3-concepts -->

An **extraction point** says what to capture during generation and which probe
receives it: *these layers*, *this tensor*, *at these token positions*, *go to
this probe*, *run it inline or async*. A **spec** is a named list of
extraction points, usually written in YAML.

## A minimal spec

```yaml
version: "1"
extraction_points:
  - name: last_prompt_token_probe
    layers: 12
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: linear_probe
    probe_kind: single_shot
    execution_mode: inline
```

This captures the residual stream at layer 12 for the last prompt token and
passes it, synchronously, to whatever probe is registered under the name
`linear_probe`.

Load it with `undercurrent.spec`:

```py
from undercurrent.spec import load_yaml_file

spec = load_yaml_file("probes.yaml")  # -> ProbeSpec
for point in spec:  # iterates ExtractionPoint objects
    print(point.name, point.tensor_type, point.layers)

point = spec.get("last_prompt_token_probe")
```

`parse_yaml(text)` and `parse_dict(data)` do the same for a YAML string or an
already-loaded dict. All three return a `ProbeSpec`: an iterable of resolved
`ExtractionPoint`s with `.get(name)`, `.names` and `len()`.

Invalid specs raise `undercurrent.spec.SpecValidationError` with a message that
names the extraction point and field at fault. Undercurrent never just warns
about an invalid spec, and never lets a raw pydantic or YAML error escape.

## Fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `name` | `str` | yes | Unique within the spec. Every `ActivationRecord` and `ProbeResult` carries it. |
| `layers` | `int` or `list[int]` | yes | Layer indices, `>= 0`. One `ActivationRecord` is produced per matched layer. |
| `tensor_type` | `residual_stream` \| `attn_out` \| `mlp_out` \| `kv` \| `final_norm` | yes | Which tensor to capture. See [Tensor types](#tensor-types). |
| `position` | int or string | yes | Which tokens to capture. See [Position selectors](position-selectors.md). |
| `stride` (alias `every_n`) | `int >= 1` | no | Keep every Nth match of a continuous `position`. Only valid for continuous selectors. |
| `until` | `generation_end` \| `fixed_count` \| `stop_token` | no | Only valid for continuous selectors. See the note below. |
| `probe_type` | `str` | yes | The name the probe is registered under. |
| `probe_kind` | `single_shot` \| `trajectory` | yes | See [Probes](probes.md#single-shot-and-trajectory-probes). |
| `probe_args` | mapping | no | Extra constructor arguments for this point's probe. They override the router-level defaults for the same `probe_type`. |
| `execution_mode` | `inline` \| `async` | no, default `inline` | See [Execution modes](execution-modes.md). |
| `queue_depth` | `int >= 1` | no | Size of this point's async queue. Only valid with `execution_mode: async`; falls back to the router's default (32) when omitted. |
| `intervention` | mapping | no | `mode`, `timeout_ms`, `on_timeout`. See [Interventions](interventions.md). When omitted, the router's `default_intervention_policy` applies. |

The top level of the file has two keys: `version` (a string, default `"1"`)
and `extraction_points`. Unknown keys anywhere are rejected.

!!! note "Older key names"
    `layer` and `tensor` are older spellings of `layers` and `tensor_type`.
    They are still accepted but emit a `DeprecationWarning`; setting both the
    old and new key on one point is an error. The serializer always writes the
    new names.

!!! note "`until` is not enforced by the shipped adapters"
    `until` is validated and round-tripped, but neither the Hugging Face nor the
    vLLM adapter reads it today. A continuous selector such as `generated[*]`
    runs until generation ends whatever `until` says. Use a bounded slice such
    as `generated[0:64]` if you need a fixed window.

### Tensor types

| Value | What it is | HF adapter | vLLM adapter |
| --- | --- | --- | --- |
| `residual_stream` | Output of a decoder layer (the hidden state after that layer) | yes | yes |
| `attn_out` | Output of the layer's attention block | yes | yes |
| `mlp_out` | Output of the layer's MLP block | yes | yes |
| `final_norm` | Output of the model's final norm, applied once after the last decoder layer. This is what `outputs.hidden_states[-1]` returns in `transformers`, and differs from `residual_stream` at the last layer. | no | yes |
| `kv` | KV cache | no | no |

An adapter raises a clear error at registration time if a spec asks for a
tensor type it can't capture. `kv` is part of the vocabulary so specs can
express it, but no shipped adapter captures it: the KV cache isn't a per-token,
per-layer forward-pass output the way the other tensors are.

To see which layers and tensor types a given model exposes, run
`undercurrent inspect-model MODEL` (see the [CLI reference](../reference/cli.md)).

## Rejected combinations

Parsing fails fast with `SpecValidationError` for:

- `execution_mode: async` with `probe_kind: single_shot`. Async execution is
  for probes that carry state across many activations; a single-shot probe has
  no state to carry.
- `queue_depth` without `execution_mode: async`.
- `stride` or `until` on a non-continuous `position` (`5`, `prompt[i]`,
  `generated[i]`). Those match exactly one token, so neither field means
  anything there.
- `execution_mode: async` with any `intervention.mode` other than `reject`.
  Async points never hold up generation, so they can't take part in a blocking
  intervention.
- An invalid `intervention`: `timeout_ms` missing (or `< 1`) under
  `block_until_signal`, or `timeout_ms` / a non-default `on_timeout` set under
  `reject`.
- `layers` empty, negative or non-integer; `stride` or `queue_depth` below 1;
  an empty `name` or `probe_type`.
- Duplicate `name`s in one spec.
- Unknown fields, and position strings that don't match the
  [grammar](position-selectors.md).

The [router](../reference/router.md) re-checks the async rules when you register
a request, so a hand-built `ExtractionPoint` that skipped the parser still
can't sneak through.

## Example specs

The repository's `examples/specs/` directory has four complete specs, each
parsed by the test suite.

**A single token, inline** (`classic_single_token.yaml`): the minimal spec
above.

**A trajectory over generation, async** (`continuous_trajectory.yaml`):
every second generated token, three layers at once, processed off the hot
path.

```yaml
version: "1"
extraction_points:
  - name: generation_trajectory
    layers: [4, 8, 12]
    tensor_type: mlp_out
    position: "generated[*]"
    stride: 2               # equivalent alias: every_n: 2
    until: generation_end
    probe_type: trajectory_probe
    probe_kind: trajectory
    execution_mode: async
    queue_depth: 8
```

**An offset position** (`offset_dual_position.yaml`): "read position *b* at
token *b+1*". `prompt[-1]+1` is one token past the last prompt token, i.e. the
first generated token.

```yaml
version: "1"
extraction_points:
  - name: read_position_at_next_token
    layers: 6
    tensor_type: attn_out
    position: "prompt[-1]+1"
    probe_type: dual_position_probe
    probe_kind: single_shot
    execution_mode: inline
```

**A fixed window** (`sliced_range.yaml`): generated tokens 5 to 19.

```yaml
version: "1"
extraction_points:
  - name: mid_generation_window
    layers: 20
    tensor_type: kv
    position: "generated[5:20]"
    probe_type: trajectory_probe
    probe_kind: trajectory
    execution_mode: inline
```

This last spec parses, but asks for `kv`, which no shipped adapter captures; it
exists to show the slice syntax. Swap in `residual_stream` to run it.

## Specs in Python

YAML is the configuration format. In Python, the resolved type is
`undercurrent.spec.ExtractionPoint`, a frozen dataclass you can build directly.
Its fields mirror the YAML keys, with `layers` always a tuple, enums for the
fixed vocabularies, and `position` already parsed into a `PositionSelector`:

```python
from undercurrent.spec import ExecutionMode, ExtractionPoint, ProbeKind, TensorType, parse_position

point = ExtractionPoint(
    name="generation_trajectory",
    layers=(4, 8, 12),
    tensor_type=TensorType.MLP_OUT,
    position=parse_position("generated[*]"),
    stride=2,
    until=None,
    probe_type="trajectory_probe",
    probe_kind=ProbeKind.TRAJECTORY,
    execution_mode=ExecutionMode.ASYNC,
    queue_depth=8,
)

# Adapters ask each point, per token, whether it should fire.
# Prompt of 10 tokens; generated tokens 0, 2, 4, ... match because of stride=2.
assert point.matches(token_index=10, is_generated=True, prompt_len=10, generated_index=0)
assert not point.matches(token_index=11, is_generated=True, prompt_len=10, generated_index=1)
assert point.matches(token_index=12, is_generated=True, prompt_len=10, generated_index=2)
assert not point.matches(token_index=3, is_generated=False, prompt_len=10)  # prompt tokens never match generated[*]
```

`ExtractionPoint.matches()` combines the position selector with `stride`.
Adapters call it; you rarely need to. Building points by hand skips the
parser's validation, but the router still re-checks the async rules at
registration.

`intervention` is `None` unless the spec set one. `None` means "use the
router's default", not "no policy".

## Round-tripping

A resolved spec serializes back to spec-shaped data:

```py
from undercurrent.spec import parse_dict, probe_spec_to_dict, to_yaml

spec = parse_dict(original_dict)
probe_spec_to_dict(spec)  # -> plain dict
to_yaml(spec)  # -> YAML string

assert parse_dict(probe_spec_to_dict(spec)) == spec
```

The round trip is lossless in meaning, though not byte-for-byte: for example a
position written as `" prompt[-1] "` comes back as `"prompt[-1]"`, and optional
fields left at their defaults are omitted. `extraction_point_to_dict(point)`
does the same for one point, which is handy for logging the config a probe ran
with.

## Validate from the command line

```bash
undercurrent validate probes.yaml
```

checks a spec without loading a model; see the [CLI reference](../reference/cli.md).

## Next

- [Position selectors](position-selectors.md): the full `position` grammar.
- [Probes & lifecycle](probes.md): what receives the activations.
- [`undercurrent.spec` API reference](../reference/spec.md).

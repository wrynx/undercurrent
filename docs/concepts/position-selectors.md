# Position selectors

<!-- owner: p3-concepts -->

A **position selector** is the `position` field of an
[extraction point](extraction-points.md). It picks which tokens of the
sequence (prompt plus generated text) an activation is captured for: one
specific token, or a continuous run of generated tokens.

```yaml
position: "prompt[-1]"      # the last prompt token
position: "generated[*]"    # every generated token
position: "generated[5:20]" # generated tokens 5..19
```

## Forms

Indices are 0-based. A position is either a bare integer or one of these
strings:

| Form | Example | Matches | Continuous? |
| --- | --- | --- | --- |
| absolute index | `5` | The token at index 5 of the full sequence (prompt + generated) | no |
| `prompt[i]` | `"prompt[0]"` | The *i*-th prompt token | no |
| `prompt[-i]` | `"prompt[-1]"` | Counting from the end of the prompt: `-1` is the last prompt token | no |
| `generated[i]` | `"generated[0]"` | The *i*-th generated token | no |
| `generated[-i]` | `"generated[-1]"` | Counting from the end of the generation (see the caveat below) | no |
| `generated[*]` | `"generated[*]"` | Every generated token | yes |
| `generated[start:]` | `"generated[5:]"` | Generated tokens from `start` onward | yes |
| `generated[start:stop]` | `"generated[2:8]"` | Generated tokens with `start <= i < stop` | yes |
| `prompt[i]` / `generated[i]` + offset | `"prompt[-1]+1"`, `"generated[3]-2"` | The base position, shifted by the offset after it is resolved | no |

Some details the table leaves out:

- An offset is a signed integer written straight after the closing bracket:
  `+1`, `-2`. No spaces inside the string (`"prompt[-1] + 1"` is rejected);
  surrounding whitespace is ignored.
- An offset can push a position across the prompt/generation boundary:
  `"prompt[-1]+1"` is the first generated token, the same token as
  `"generated[0]"`.
- Bare integers don't take offsets (`"5+1"` is rejected); write `6`.
- Slices need an explicit, non-negative start and a `stop` greater than
  `start`. `"generated[:5]"`, `"generated[-3:]"` and `"generated[3:3]"` are
  rejected.
- There is no wildcard or slice over the prompt (`"prompt[*]"`,
  `"prompt[0:4]"`). To capture several prompt tokens, use one extraction point
  per position.
- Anything that doesn't fit the grammar fails spec parsing with a
  `SpecValidationError` that lists the accepted forms.

## Continuous selectors

`generated[*]` and the two slice forms are **continuous**: they match many
tokens, one after another, as generation proceeds. Only continuous selectors
accept `stride` (alias `every_n`) and `until`:

```yaml
position: "generated[*]"
stride: 4          # generated tokens 0, 4, 8, ...
```

On a slice, the stride counts from the slice's start: `"generated[5:]"` with
`stride: 2` matches generated tokens 5, 7, 9, …

A continuous selector pairs naturally with a `trajectory` probe, which keeps
state across all the activations it receives (see [Probes](probes.md)).

!!! warning "Offsets on continuous selectors have no effect"
    The parser accepts an offset after `generated[*]` or a slice (for example
    `"generated[*]+1"`), but matching ignores it. Shift the slice bounds
    instead: `"generated[1:]"`.

## Caveats

!!! warning "`generated[-1]` doesn't match during streaming"
    A negative generated index can only be resolved once the total number of
    generated tokens is known, and that isn't known until generation ends. The
    shipped Hugging Face and vLLM adapters match positions while generating, so
    `generated[-1]` (and any `generated[-i]`) never fires with them today. To
    look at the end of a generation, use a continuous selector with a
    `trajectory` probe and keep the last activation it saw.

Other things to know:

- A position past the end of the actual sequence, for example `"prompt[50]"`
  on a 10-token prompt, simply never matches. It is not an error.
- A negative absolute index such as `-1` parses but never matches. Use
  `"prompt[-1]"` or a generated form.
- The adapters capture an activation when a token is *fed through* the model.
  The first generated token is processed at the first decode step after the
  prompt, so its record has `token_pos == prompt_len` and
  `is_generated=True`.

## How selectors are evaluated

Parsing turns the string into a `PositionSelector`, a small frozen value.
Adapters call its `matches()` method for every token; they never re-parse the
string:

```python
from undercurrent.spec import PositionKind, parse_position

prompt_len = 10

last_prompt = parse_position("prompt[-1]")
assert last_prompt.kind is PositionKind.PROMPT_INDEX
assert last_prompt.matches(token_index=9, is_generated=False, prompt_len=prompt_len)

# One past the last prompt token is the first generated token.
next_token = parse_position("prompt[-1]+1")
assert next_token.resolve_absolute(prompt_len) == 10
assert next_token.matches(token_index=10, is_generated=True, prompt_len=prompt_len, generated_index=0)

# A slice matches by index within the generated text.
window = parse_position("generated[2:5]")
matched = [g for g in range(8) if window.matches(prompt_len + g, True, prompt_len, generated_index=g)]
assert matched == [2, 3, 4]
assert window.is_continuous

# generated[-1] needs the final length, which streaming adapters don't have.
last_generated = parse_position("generated[-1]")
assert not last_generated.matches(12, True, prompt_len, generated_index=2)
assert last_generated.matches(12, True, prompt_len, generated_index=2, num_generated_total=3)
```

The arguments to `matches()`:

| Argument | Meaning |
| --- | --- |
| `token_index` | Absolute index of the token in the full sequence |
| `is_generated` | Whether the token is part of the generation (not the prompt) |
| `prompt_len` | Number of prompt tokens |
| `generated_index` | Index within the generated part; needed for continuous selectors |
| `num_generated_total` | Total generated length; needed only for negative `generated[i]` |

`PositionSelector.to_raw()` turns a selector back into the YAML value it came
from; the spec serializer uses it for [round-tripping](extraction-points.md#round-tripping).

## Next

- [Probes & lifecycle](probes.md)
- [`undercurrent.spec` API reference](../reference/spec.md)

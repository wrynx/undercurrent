# Quickstart

<!-- owner: p3-quickstart -->

In five minutes you will attach a probe to GPT-2, read its result, make a
probe stop a generation part-way through, and log a probe's output to a file
without touching the response. Everything runs on a laptop CPU.

You need Undercurrent installed (`pip install undercurrent`, see
[Installation](installation.md)) and about 500 MB of disk for GPT-2, which is
downloaded from the Hugging Face Hub the first time you load it.

The Python blocks on this page form one script: run them in order in one
interpreter or notebook, because later blocks use names from earlier ones.

## 1. Find a layer to probe

Before writing a spec, look at what the model offers. `undercurrent
inspect-model` reads only the model's `config.json`, so it is instant and
downloads no weights:

```console
$ undercurrent inspect-model gpt2 --backend hf
model:       gpt2
class:       GPT2LMHeadModel (model_type=gpt2)
layers:      12
hidden size: 768
heads:       12
vocab:       50257
weights:     not loaded (structure built on the meta device)

tensor_type      hf
residual_stream  yes
attn_out         yes
mlp_out          yes
kv               no [1]
final_norm       no [2]
  [1] the KV cache isn't a per-layer forward-pass tensor with one row per token; the HF adapter doesn't capture it
  [2] the HF adapter doesn't hook the model-level final norm (the vLLM adapter does)

Probe it with layers 0..11, e.g. layers: [6], position: "prompt[-1]"
```

GPT-2 has 12 layers, so layer 6 is in the middle. The residual stream is the
running hidden state between layers, a 768-dimensional vector per token.

## 2. Write a spec

A spec lists **extraction points**: which tensor to capture, at which layers
and token positions, and which probe receives it. This one captures the
residual stream at layer 6 for the last prompt token and hands it to a probe
called `norm_gate`, which runs `inline`, on the generation path:

```python
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
```

A spec can also be a YAML file path, or a list of `ExtractionPoint(...)`
objects built in Python. [Extraction points & specs](../concepts/extraction-points.md)
covers every field, and [Position selectors](../concepts/position-selectors.md)
covers `position`.

## 3. Write a probe

A probe is the code that looks at each captured activation. The simplest kind
is a plain function decorated with `@probe`. It receives an
`ActivationRecord` and returns a score:

```python
from undercurrent.core import ActivationRecord, probe


@probe("norm_gate", threshold=500.0)
def norm_gate(record: ActivationRecord) -> float:
    """L2 norm of the residual stream at this token."""
    return float(record.tensor.norm())
```

A score at or above `threshold` would stop generation. GPT-2's norms here are
far below 500, so this probe only records the score.

## 4. Generate

`ProbedModel.from_pretrained` loads the model, checks the spec against it and
attaches the probes. `generate` runs the model and returns the text together
with every probe's result:

```python
from undercurrent.model import ProbedModel

PROMPT = "The quick brown fox"

with ProbedModel.from_pretrained("gpt2", spec=SPEC, probes={"norm_gate": norm_gate}) as model:
    out = model.generate(PROMPT, max_new_tokens=20, temperature=0)

print(repr(out.text))
print(out.probe_results["last_prompt_token"].verdict)
assert out.aborted is False
```

You should see 20 tokens of GPT-2 text and a verdict like
`{'flagged': False, 'max_score': 106.25..., 'n': 1}`: the probe saw one
activation, with a norm of about 106, and didn't flag it.
`temperature=0` means greedy decoding, so the text is the same every run.

`out` is a `GenerationOutput`. Besides `text` and `probe_results` it has
`aborted`, `abort_reason` and `abort_point`, which the next step uses.

## 5. Stop a generation

Now watch every generated token instead of one prompt token, and stop the
generation from inside the probe. A probe that decides from a sequence of
activations is a **trajectory** probe. Trajectory probes keep state between
activations, so they are written as a class. This one tracks the mean norm of
the generated tokens and aborts once it has seen `min_tokens` tokens with a
mean of at least `threshold`:

```python
from undercurrent.core import Probe, ProbeAction, ProbeResult, ProbeSignal, RequestContext


class MeanNormProbe(Probe):
    probe_kind = "trajectory"

    def __init__(self, threshold: float = 150.0, min_tokens: int = 5) -> None:
        super().__init__()
        self.threshold = threshold
        self.min_tokens = min_tokens

    def on_start(self, request_ctx: RequestContext) -> None:
        self.norms: list[float] = []
        self.signals: list[ProbeSignal] = []

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        self.norms.append(float(record.tensor.norm()))
        mean = sum(self.norms) / len(self.norms)
        if self.signals or len(self.norms) < self.min_tokens or mean < self.threshold:
            return None
        signal = ProbeSignal(action=ProbeAction.ABORT, confidence=mean, metadata={"mean_norm": mean})
        self.signals.append(signal)
        return signal

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        mean = sum(self.norms) / len(self.norms) if self.norms else None
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"mean_norm": mean, "tokens": len(self.norms)},
            signal_history=list(self.signals),
        )
```

The spec points it at every generated token (`generated[*]`) and sets a low
threshold through `probe_args`, which are passed to the probe's constructor.
GPT-2's layer-6 norms are well above 50, so the probe aborts as soon as it
has seen 5 tokens:

```python
DRIFT_SPEC = """
version: "1"
extraction_points:
  - name: drift
    layers: 6
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: mean_norm
    probe_kind: trajectory
    execution_mode: inline
    probe_args:
      threshold: 50.0
      min_tokens: 5
"""

with ProbedModel.from_pretrained("gpt2", spec=DRIFT_SPEC, probes={"mean_norm": MeanNormProbe}) as model:
    out = model.generate(PROMPT, max_new_tokens=20, temperature=0)

print(repr(out.text))
print("aborted:", out.aborted)
print(out.abort_reason)
print(out.probe_results["drift"].verdict)
assert out.aborted is True
assert out.abort_point == "drift"
```

The generation stops after 5 tokens instead of 20, `out.aborted` is
`True`, and `out.abort_reason` names the extraction point that stopped it. Because the extraction point is `inline`, the
probe's `ABORT` signal stops the model before the next token.
[Interventions](../concepts/interventions.md) explains what else a probe can
do, and [Write a custom probe](../guides/custom-probe.md) builds probes like
these step by step.

## 6. Collect results and log them

**In-process callback.** Pass `on_result=` to receive every
`GenerationOutput` as it finishes, for example to collect them in a list
while you generate a batch of prompts:

```python
collected = []

with ProbedModel.from_pretrained(
    "gpt2", spec=DRIFT_SPEC, probes={"mean_norm": MeanNormProbe}, on_result=collected.append
) as model:
    model.generate(["Once upon a time", "The weather today is"], max_new_tokens=10, temperature=0)

for output in collected:
    print(repr(output.prompt), "aborted:", output.aborted, output.probe_results["drift"].verdict)
assert len(collected) == 2
```

**Observe mode.** An inline probe runs on the generation path, so it can
stop a generation but also adds its own time to every token. In production
you often only want to watch. Switch the same probe to
`execution_mode: async` and it runs on a background worker, off the
generation path. Its signals become observations: the generation is never
stopped, and the signals and the final result go to a log sink. Here a
`FileLogSink` writes them as newline-delimited JSON:

```python
import json
import tempfile
from pathlib import Path

from undercurrent.sinks import FileLogSink

OBSERVE_SPEC = DRIFT_SPEC.replace("execution_mode: inline", "execution_mode: async")
log_path = Path(tempfile.mkdtemp()) / "observations.ndjson"

with ProbedModel.from_pretrained(
    "gpt2", spec=OBSERVE_SPEC, probes={"mean_norm": MeanNormProbe}, log_sink=FileLogSink(log_path)
) as model:
    out = model.generate(PROMPT, max_new_tokens=20, temperature=0)

print("aborted:", out.aborted)  # observe mode never stops the generation
assert out.aborted is False
for line in log_path.read_text().splitlines():
    record = json.loads(line)
    print(record["kind"], record["extraction_point_name"], record["payload"].get("verdict", record["payload"]))
```

Each line of the file is a JSON object with `kind`, `request_id`,
`extraction_point_name`, `timestamp` and `payload`. Here there is one
`signal` line (the abort the probe asked for, recorded but not acted on) and
one `result` line whose payload holds the final verdict. The probe saw almost
the whole generation this time, because nothing stopped it. In
production the same sink can be a `WebhookLogSink` with retries, dead-lettering
and redaction, and the async workers have bounded queues and overflow
policies. That is the [Production](../guides/vllm-deployment.md) section.

## Next steps

- [Architecture](../concepts/architecture.md): how specs, the router,
  probes and engine adapters fit together.
- [Write a custom probe](../guides/custom-probe.md): function and class
  probes, testing them without a model, and shipping them as a plugin.
- Production: [Deploy with vLLM](../guides/vllm-deployment.md) and
  [Embed in your serving stack](../production/embedding.md).
- [Examples](../examples.md): complete, runnable projects.
- [API reference](../reference/index.md): every public class and function.

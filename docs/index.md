---
title: Undercurrent
---

# ![Undercurrent](assets/brand/undercurrent-logo.svg#only-light){ width="420" } ![Undercurrent](assets/brand/undercurrent-logo-dark.svg#only-dark){ width="420" }

<!-- owner: p3-quickstart -->

**Wrynx's activation-probing platform.** *See the undercurrent before it surfaces.*

Undercurrent is production-grade activation probing for live inference: it
reads a model's internal activations while it generates, runs your probes on
them, and can stop a generation the moment a probe objects.

- **Declarative specs.** A short YAML spec says which tensor to capture, at
  which layers and token positions, and which probe receives it. No model
  surgery, no forked serving code.
- **Inline intervention.** Probes run on the generation path and can abort a
  generation mid-stream, before the next token is produced.
- **Production-ready async execution on vLLM.** Observe-only probes run off the
  hot path on bounded worker queues with overflow policies, and report to
  file, webhook and metrics sinks.

```py
from undercurrent.core import probe
from undercurrent.model import ProbedModel


@probe("norm_gate", threshold=500.0)
def norm_gate(record):
    return float(record.tensor.norm())


with ProbedModel.from_pretrained("openai-community/gpt2", spec=SPEC, probes={"norm_gate": norm_gate}) as model:
    out = model.generate("The quick brown fox", max_new_tokens=20, temperature=0)
print(out.text, out.probe_results)
```

`SPEC` is a few lines of YAML; the [Quickstart](getting-started/quickstart.md)
walks through it end to end on a laptop CPU.

<div class="grid cards" markdown>

-   **Getting started**

    ---

    Install with `pip install undercurrent` and run your first probe on GPT-2.

    [Installation](getting-started/installation.md) ·
    [Quickstart](getting-started/quickstart.md)

-   **Concepts**

    ---

    Extraction points, position selectors, probe kinds, and inline vs async
    execution.

    [Architecture](concepts/architecture.md)

-   **Guides**

    ---

    Write, test and package your own probe.

    [Write a custom probe](guides/custom-probe.md)

-   **Production**

    ---

    Run probes in vLLM, embed the router in your own serving stack, tune
    queues, timeouts, sinks and metrics.

    [Deploy with vLLM](guides/vllm-deployment.md) ·
    [Embed in your serving stack](production/embedding.md)

-   **API reference**

    ---

    Every public class and function, and the `undercurrent` CLI.

    [API reference](reference/index.md)

</div>

# Intended use & limitations

<!-- owner: p3-intended-use -->

!!! warning "Undercurrent is machinery, not a safety product"
    Undercurrent runs probes on a model's activations during inference. It can
    observe them, log what they report, and abort generation when they say so.
    It ships **no trained classifiers**. Any safety decision is only as good as
    the probes you bring.

    The content-safety code in
    [`examples/content_safety/`](https://github.com/wrynx/undercurrent/tree/main/examples/content_safety)
    is a **wiring demo**. Its probes use random or dummy weights, its scores
    mean nothing, and it is not installed with the package. It is **not** a
    validated safety classifier.

## Responsible use

Undercurrent gives you hooks inside a running model. Use them to understand
and monitor models, not to watch people. Before you rely on a probe, check
how well it works on data like yours. Don't use the bundled demos or any
untested probe as a production filter, and never let one probe be your only
safety control. Tell your users what you log about them, and keep it only as
long as you need it (see [Privacy & data handling](privacy.md)). Report
security problems privately, as described in
[`SECURITY.md`](https://github.com/wrynx/undercurrent/blob/main/SECURITY.md).
The rest of this page explains each point.

## What Undercurrent is for

- **Runtime monitoring.** Watch what a deployed model is doing internally,
  request by request. Log probe verdicts to a file or webhook, and track
  per-probe queue depth, drops, latency and errors.
- **Safety research.** Study how concepts, refusals or harmful intent appear
  in activations, on Hugging Face Transformers models or in a vLLM
  deployment, without forking the inference engine.
- **Interpretability-informed guardrails.** Run a probe you have trained and
  evaluated inline, and let it flag or abort a generation as one layer of a
  defence-in-depth setup.
- **Evaluation.** Score models, prompts or datasets with probes during
  generation. For example, measure how often a probe fires on a benchmark,
  or compare layers and token positions.

## Out of scope and discouraged uses

- **Using the content-safety demo, or any untrained or untested probe, as a
  production safety filter.** The demo shows how the parts connect. A probe
  with random weights blocks or allows content at random.
- **Relying on a single probe as your only safety control.** Probes miss
  things and can be evaded. Combine them with other measures: input and
  output filters, rate limits, human review and abuse monitoring.
- **Covert surveillance of end users.** Don't use probe verdicts or logged
  metadata to profile, track or make decisions about individuals without
  telling them. Don't use them to infer sensitive traits (health, religion,
  sexual orientation, political views and similar) from what people write.
  Tell users what you monitor, and follow the law where you operate.

## Limitations

- **Probe accuracy depends on training data.** A probe learns whatever
  separates its training examples, which may be a shortcut (topic, length,
  formatting) rather than the concept you care about. Its precision and recall
  hold only for data like the data it was evaluated on.
- **Distribution shift.** Probes trained on one model, checkpoint, layer,
  prompt template, language or domain can degrade silently on another. A
  fine-tune or a new system prompt can change activations enough to break a
  probe. Re-evaluate after any change to the model or how you use it.
- **Adversarial robustness is unknown.** Undercurrent makes no claims about how
  well any probe resists deliberate evasion. Assume a motivated attacker can
  find inputs that keep a probe's score low. Red-team your probes before you
  rely on them.
- **Latency impact of inline probes.** Inline probes run on the generation
  path. Every inline probe adds its own compute to each token it sees, and
  `block_until_signal` policies can wait up to their `timeout_ms`. On vLLM,
  that wait blocks every request batched in the same step, not just the one
  being probed. Use async (observe-only) execution where you don't need to
  intervene, and measure latency under load. See
  [Execution modes](../concepts/execution-modes.md) and
  [Intervention policies](../production/intervention-policies.md).
- **Engine-version coupling.** The vLLM adapter hooks vLLM internals that
  aren't a public API, so each Undercurrent release supports a narrow range
  of vLLM versions, which the adapter checks at runtime. Upgrading vLLM can
  break capture until Undercurrent is updated. See
  [Compatibility](../compatibility.md).

## Dual-use note

Activation access is dual-use. The same hooks that let you detect harmful
intent can also be used to study how a model's safeguards work, and to look
for ways to weaken or bypass them.

What the project does:

- Provides the machinery to read activations and run probes during
  inference, on models you already have access to.
- Lets probes **observe**, **flag** and **abort** generation.

What the project does not do:

- It doesn't modify activations or weights. There is no steering, ablation or
  "refusal removal" feature, and none is planned for v0.1.
- It doesn't ship trained probes, jailbreaks or recipes for disabling a
  model's safeguards.
- It doesn't give access to a model you couldn't already run.

We follow responsible-disclosure norms. If you find a vulnerability in
Undercurrent, or a way it can be misused to cause serious harm, report it
privately as described in
[`SECURITY.md`](https://github.com/wrynx/undercurrent/blob/main/SECURITY.md)
(email [security@wrynx.com](mailto:security@wrynx.com)), not in a public
issue. If your research shows how to weaken a third-party model's safeguards,
tell that model's developer before you publish.

## Training and sharing probes

If you train a probe, and especially if you share it, document it the way you
would document a model (a "probe card"). At a minimum, include:

- **Intended use.** What the probe detects, and what it shouldn't be used
  for.
- **Target model and extraction point.** The exact model and revision, the
  layer, the tensor type and the token positions, ideally as the YAML spec
  you used.
- **Training data.** Where the positive and negative examples came from, how
  they were labelled, their size, languages and domains, and any known gaps
  or biases.
- **Evaluation.** Metrics on a held-out set (for example precision, recall,
  false-positive rate at your chosen threshold, and AUROC), the threshold
  itself, and how the evaluation data differs from the training data.
- **Known failure modes.** Inputs where it misfires, sensitivity to prompt
  format or language, any red-teaming you did, and what you didn't test.
- **Operational notes.** Inline or async, measured latency overhead, and the
  Undercurrent and engine versions you tested with.

Publishing pretrained probes on the Hugging Face Hub is on the roadmap. It is
not part of v0.1.

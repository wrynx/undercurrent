# Comparison with related tools

<!-- owner: p3-comparison -->

Undercurrent runs activation probes inside an inference engine while it serves
requests. A declarative spec says which activations to read. Each request gets
its own probe instances, and a probe can either stop that request mid-generation
or report asynchronously without blocking it. It is not a research
interpretability toolkit, and it does not replace text-level guardrails. Most
teams that use it will also use at least one of the tools below.

## When to use Undercurrent, and when to use something else

**Undercurrent is a good fit when:**

- You already have a probe (a linear classifier on the residual stream, for
  example) and want it to run on every request a model serves, not in a
  notebook.
- A probe should be able to stop a generation as soon as it fires, rather than
  after the full response has been produced and checked.
- Probe results should flow to files, webhooks or metrics without slowing down
  generation. The router runs probes on bounded per-binding queues with worker
  pools and overflow policies (see [Async execution](production/async-execution.md)).
- You want the probing configuration in a reviewable YAML file
  ([extraction points](concepts/extraction-points.md)) rather than in hook code.

**Use something else when:**

- **You are doing interpretability research** (activation patching, circuit
  analysis, attribution, editing activations and studying the effect).
  TransformerLens, nnsight and pyvene have far richer tooling for this.
  Undercurrent reads activations and runs probes; it does not patch or edit
  them.
- **You need to train a probe.** Collect activations and train with a research
  tool or plain PyTorch, then deploy the probe with Undercurrent (see
  [Using them together](#using-them-together)).
- **You want to experiment on a model too large for your hardware.** nnsight
  with NDIF runs your code against remotely hosted models. Undercurrent runs
  only on hardware you control.
- **Your model is behind an API you don't host** (a hosted chat API, for
  example). Undercurrent needs access to the model's forward pass. Text-level
  tools such as NeMo Guardrails or a Llama Guard classifier work on any model's
  inputs and outputs.
- **You need content-safety classification with a published hazard taxonomy.**
  Llama Guard is a trained, documented safety classifier. Undercurrent ships no
  pretrained probes; you bring your own.
- **You need dialogue control**: topical rails, scripted flows, retrieval or
  tool-call checks. That is what NeMo Guardrails is for.
- **You run vLLM with multiple workers.** The Undercurrent vLLM adapter is
  validated with exactly one model worker. Tensor-parallel results from several
  workers are merged and deduplicated, but the adapter refuses to start with
  more than one worker unless you override the check, and pipeline parallelism
  and Ray/distributed executors are not validated (see
  [Tensor & pipeline parallelism](internals/vllm-parallelism.md)). It also
  supports only the vLLM minor listed on the
  [Compatibility](compatibility.md) page (currently `>=0.28,<0.29`).
- **You need batched generation with the HF backend.** The Hugging Face
  transformers adapter handles one `generate()` call at a time with batch size 1
  and no beam search. Use the vLLM adapter for concurrent requests.

## Comparison table

Legend: ✓ supported and documented · partial (see footnote) · ✗ not supported or
not a goal of the project · n/a the question doesn't apply to this kind of tool ·
unclear: we could not verify it from the project's documentation.

| Tool | Purpose | Works inside vLLM serving | Inline intervention (abort) | Async observation | Per-request isolation | Declarative spec | Research ergonomics | License |
|---|---|---|---|---|---|---|---|---|
| **Undercurrent** 0.1 | Production activation probing | partial[^uc-vllm] | ✓[^uc-abort] | ✓ | ✓[^uc-iso] | ✓ (YAML) | partial[^uc-research] | Apache-2.0 |
| **TransformerLens** 4.0.0 | Mechanistic interpretability | ✗[^tl-vllm] | partial[^research-abort] | ✗ | n/a | ✗ | ✓ | MIT |
| **nnsight** 0.7.0 | Interpretability; access to and editing of model internals, local or remote (NDIF) | ✓[^nn-vllm] | partial[^research-abort] | partial[^nn-async] | ✓[^nn-iso] | ✗ (Python tracing code) | ✓ | MIT |
| **pyvene** 0.1.8 | Interventions on model internals (interpretability, steering, editing) | ✗[^pv-vllm] | partial[^research-abort] | ✗ | n/a | partial[^pv-spec] | ✓ | Apache-2.0 |
| **baukit** (unversioned, GitHub) | Research prototyping utilities, including activation tracing | ✗[^bk-vllm] | partial[^research-abort] | ✗ | n/a | ✗ | ✓ | MIT[^bk-license] |
| **Llama Guard 4** (12B) | Safety classifier for prompts and responses (text and images) | ✓ as a separate model[^lg-vllm] | n/a[^lg-abort] | n/a | n/a | n/a | n/a | Llama 4 Community License |
| **NeMo Guardrails** 0.24.1 | Programmable guardrails for LLM conversational systems | n/a[^nemo-vllm] | partial[^nemo-abort] | unclear | unclear | ✓ (YAML + Colang) | n/a | Apache-2.0 |
| **vLLM built-ins** 0.30.0 (logits processors, hidden-state extraction) | Extension points of the serving engine itself | ✓ | unclear[^vllm-abort] | partial[^vllm-hs] | partial[^vllm-iso] | partial[^vllm-spec] | ✗ | Apache-2.0 |

Versions are the latest release on PyPI (or the latest model release for Llama
Guard) on the review date below; see [Sources](#sources).

[^uc-vllm]: One model worker is the validated configuration. The adapter refuses
    to start with more than one worker unless you pass
    `allow_unsupported_executor=True`. Tensor-parallel results are merged across
    workers, but pipeline parallelism and Ray/distributed executors are not
    validated. Only vLLM's V1 model runner is supported, KV-cache extraction
    points are rejected, and negative generated-token selectors such as
    `generated[-1]` never match under vLLM. See
    [Tensor & pipeline parallelism](internals/vllm-parallelism.md) and
    [Compatibility](compatibility.md).
[^uc-abort]: Inline probes can return an abort signal that stops generation for
    that request (HF: a stopping criterion; vLLM: request abort). The
    `block_until_signal` intervention policy waits for a probe's verdict with a
    timeout. With vLLM it holds up the whole batched step, so it suits
    single-request deployments; see [Interventions](concepts/interventions.md).
[^uc-iso]: The router creates a fresh probe instance per request and extraction
    point, and stores probe state keyed by `(request_id, extraction_point_name)`.
[^uc-research]: You can read activations at chosen layers, tensor types and
    token positions, but there is no activation patching or editing, no
    visualisation, and the HF backend runs one sequence at a time.
[^tl-vllm]: TransformerLens runs models through its own `TransformerBridge`
    (which replaced `HookedTransformer.from_pretrained` in 4.0). Its README
    documents no vLLM integration.
[^research-abort]: Activations can be read and edited during forward passes,
    including during generation. Stopping one request mid-generation from a
    probe's verdict is not a documented feature; you would write your own
    generation loop.
[^nn-vllm]: `nnsight.modeling.vllm.VLLM` wraps vLLM, including continuous
    batching, tensor parallelism, `AsyncLLM` streaming and an `nnsight-serve`
    FastAPI server. The nnsight team notes that it forces eager mode, bypassing
    CUDA graphs. It needs vLLM >= 0.12 and nnsight >= 0.5.13.
[^nn-async]: With `AsyncLLM`, results are streamed through an async iterable.
    Intervention code runs in the forward pass; we found no documented option
    to run it off the generation path on worker queues.
[^nn-iso]: The nnsight × vLLM blog post describes "one prompt per invoke and one
    mediator per request", with each request carrying its own intervention code.
[^pv-vllm]: pyvene works with PyTorch models in-process; its README documents no
    vLLM integration.
[^pv-spec]: Interventions are specified as dicts and can be saved and shared
    through Hugging Face as serialisable objects. Probe execution and deployment
    policy are outside its scope.
[^bk-vllm]: baukit's `Trace` and `TraceDict` hook named PyTorch modules. Its
    README documents no vLLM integration. The last commit is from 2024-02-22 and
    it is installed from GitHub (it is not on PyPI).
[^bk-license]: The repository has no LICENSE file. `setup.cfg` declares the
    classifier `License :: OSI Approved :: MIT License`.
[^lg-vllm]: Llama Guard is a separate classifier model. Its model card documents
    inference with Transformers, vLLM and SGLang. It reads text and images,
    not another model's activations.
[^lg-abort]: Llama Guard outputs "safe" or "unsafe" plus the violated
    categories. Whether and when to stop a generation is up to the application
    that calls it.
[^nemo-vllm]: NeMo Guardrails wraps calls to an LLM and works on messages and
    text; it does not run inside the inference engine.
[^nemo-abort]: With streaming output rails, the stream ends with an error object
    when a rail blocks a chunk. Rails check text in chunks of `chunk_size`
    tokens (default 200), not activations at each token.
[^vllm-abort]: Logits processors see only the logits tensor and batch layout,
    not hidden states. We found no documented way for a logits processor to
    abort a request.
[^vllm-hs]: The hidden-state extraction feature (vLLM >= 0.18) saves prompt-token
    hidden states for selected layers to a safetensors file per request. It is
    designed for training speculative-decoding draft models, and generated
    tokens are not captured.
[^vllm-iso]: Logits processors can receive per-request custom arguments, but
    they run at batch granularity on the whole step.
[^vllm-spec]: Logits processors and hidden-state extraction are configured with
    engine arguments and entry points. There is no spec format for probes.

## What each tool is good at

**TransformerLens.** A library for mechanistic interpretability. You can cache
any internal activation and add functions that edit, remove or replace
activations as the model runs. Version 4.0 made `TransformerBridge` the
recommended API; the README says it supports more than 15,000 models across
more than 140 architecture families. It has a large ecosystem of tutorials and
is a common starting point for circuit-style research.

**nnsight.** Lets you read, edit and save values inside any PyTorch model during
a forward pass, without registering hooks or refactoring the model. It ships
wrappers for Hugging Face, diffusers and vLLM, batches several prompts per trace
(each `invoke` sees only its own rows), and steps through generation with
`tracer.iter`. Through NDIF, the same trace can run remotely on a model you
can't host yourself. Its vLLM runtime brings arbitrary Python interventions to
continuous batching and tensor parallelism.

**pyvene.** A library for intervening on the internal states of PyTorch models
for model editing, steering, robustness and interpretability. Interventions are
specified as dicts, so they can be saved and shared through Hugging Face. They
work on any PyTorch model (the README mentions RNNs, ResNets, CNNs and Mamba)
and on decoding steps of generative language models.

**baukit.** David Bau's toolkit for quick research prototyping. Its `Trace` and
`TraceDict` context managers read and alter the outputs of named modules with
very little code. It also includes notebook widgets and running-statistics
utilities.

**Llama Guard.** Meta's safety classifier. Llama Guard 4 is a 12B natively
multimodal model that classifies prompts and responses against a 14-category
hazard taxonomy aligned with MLCommons. It works with any model's text, so it
is a natural choice when you don't control the generator or want a trained,
documented content policy.

**NeMo Guardrails.** NVIDIA's toolkit for adding programmable guardrails to
LLM-based conversational systems. It provides input, dialog, retrieval,
execution and output rails, configured with YAML and the Colang modelling
language. It works with many LLM providers and can check streaming output in
chunks.

**vLLM's built-in extension points.** Custom logits processors run inside vLLM
on every step, can take per-request arguments, and can be registered by class
name, CLI flag or entry point. The hidden-state extraction feature writes
prompt hidden states to disk for training speculative-decoding models. If all
you need is to change the logits or dump prompt-token hidden states, these
avoid adding a dependency.

## Using them together

- **Train with a research tool, deploy with Undercurrent.** Collect residual-stream
  activations on a labelled dataset with TransformerLens, nnsight or baukit, train
  a linear probe in PyTorch, then wrap it as an Undercurrent probe (see
  [Writing a custom probe](guides/custom-probe.md)) and point an extraction
  point at the same layer and token position. Check that the layer index and
  tensor type mean the same thing in both tools, because naming conventions
  differ.
- **Activation probes and text classifiers side by side.** Run inline
  Undercurrent probes to stop a generation early when an internal signal fires,
  and run Llama Guard on the final prompt and response as an independent
  text-level check. The two use different evidence, so they can catch different
  failures.
- **Undercurrent inside a guardrailed application.** NeMo Guardrails controls
  the dialogue and text-level policy around the LLM call. Undercurrent runs
  inside the vLLM or HF engine that serves that call and reports activation-level
  signals to your sinks.
- **Investigate with nnsight, monitor with Undercurrent.** When a production
  probe flags requests, replay them with nnsight (locally or on NDIF) to study
  what the model was doing at those layers.

## Sources

Versions, release dates and licenses were checked on the review date below.

- **TransformerLens**: [PyPI `transformer-lens`](https://pypi.org/project/transformer-lens/)
  (4.0.0, released 2026-09-21; a `v4.1.0` GitHub release tagged 2026-09-28
  was not yet on PyPI); [GitHub README](https://github.com/TransformerLensOrg/TransformerLens)
  (purpose, `TransformerBridge`, model coverage, MIT license).
- **nnsight**: [PyPI `nnsight`](https://pypi.org/project/nnsight/) (0.7.0,
  released 2026-05-05; 0.8.0rc1 pre-release 2026-09-09; MIT);
  [GitHub README](https://github.com/ndif-team/nnsight) (purpose, backends,
  NDIF `remote=True`, `invoke` batching, `tracer.iter`);
  [vLLM support docs](https://nnsight.net/documentation/modeling/vllm/);
  [NNsight × vLLM: Interpretability at Production Scale](https://nnsight.net/blog/2026/07/13/nnsight--vllm-interpretability-at-production-scale/)
  (`nnsight-serve`, continuous batching, `AsyncLLM`, one mediator per request,
  eager mode); [NNsight 0.5.13 release notes](https://discuss.ndif.us/t/nnsight-0-5-13-release-vllm-integration-and-performance-improvements/128).
- **pyvene**: [PyPI `pyvene`](https://pypi.org/project/pyvene/) (0.1.8,
  released 2025-05-26; Apache-2.0); [GitHub README](https://github.com/stanfordnlp/pyvene)
  (purpose, dict-specified interventions, supported models).
- **baukit**: [GitHub repository](https://github.com/davidbau/baukit) (README,
  last commit 2024-02-22, not on PyPI);
  [`setup.cfg`](https://github.com/davidbau/baukit/blob/main/setup.cfg) (MIT
  classifier).
- **Llama Guard 4**: [model card on Hugging Face](https://huggingface.co/meta-llama/Llama-Guard-4-12B)
  (purpose, hazard taxonomy, output format, supported inference frameworks,
  Llama 4 Community License); [PurpleLlama repository](https://github.com/meta-llama/PurpleLlama).
- **NeMo Guardrails**: [PyPI `nemoguardrails`](https://pypi.org/project/nemoguardrails/)
  (0.24.1, released 2026-09-16; Apache-2.0);
  [GitHub README](https://github.com/NVIDIA-NeMo/Guardrails) (purpose, rail
  types, Colang, LLM providers);
  [Output rail streaming configuration](https://docs.nvidia.com/nemo/guardrails/configure-guardrails/yaml-schema/streaming/output-rail-streaming)
  (`chunk_size`, `stream_first`, behaviour when a rail blocks).
- **vLLM**: [PyPI `vllm`](https://pypi.org/project/vllm/) (0.30.0, released
  2026-09-22; Apache-2.0);
  [Custom logits processors](https://docs.vllm.ai/en/latest/features/custom_logitsprocs/);
  [Extracting hidden states from vLLM](https://vllm.ai/blog/2026-03-30-extract-hidden-states)
  (prompt tokens only, safetensors output, vLLM >= 0.18).
- **Undercurrent**: claims are based on the source in this repository,
  specifically `undercurrent/adapters/vllm/adapter.py` (worker-topology check,
  per-worker result merging), `undercurrent/adapters/hf/adapter.py` (batch size
  1, no beam search) and `undercurrent/router/` (per-request probe instances,
  async queues and overflow policies).

**Last reviewed: 2026-10-01.** All of these tools change quickly, and some cells
may be out of date by the time you read this. If you spot an error or an
unfair characterisation, please
[open an issue](https://github.com/wrynx/undercurrent/issues).

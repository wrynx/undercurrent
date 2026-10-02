# Roadmap

This is where Undercurrent is heading after v0.1. It is **directional, not a
commitment**: priorities change as we learn what users need, and nothing here
has a date. To suggest a change, start a thread in
[Discussions](https://github.com/wrynx/undercurrent/discussions).

- **Now**: the next things maintainers intend to work on.
- **Next**: planned, usually building on something in *Now*.
- **Later**: wanted, not yet scheduled.

Items marked **help wanted** are good places for outside contributors to make
a big difference. Comment on the related issue or Discussion before starting,
so work isn't duplicated.

## Now

- **OpenAI-compatible `undercurrent serve` server.** A built-in server that
  returns probe verdicts alongside completions. Today,
  [`examples/openai_server/`](examples/openai_server/) is a reference wire
  format and HTTP server you can adapt.
- **Probe save/load and push to the Hugging Face Hub.** Save trained probe
  weights and configuration, load them back, and share them on the Hub.
- **vLLM tensor parallelism (one pipeline stage).** The vLLM adapter
  currently supports a single model-forward worker and refuses multi-worker
  executors. The research notes find that, for stock Llama-style models,
  every TP rank already sees the full tensor; the remaining work is
  emitting each activation once instead of once per rank.

## Next

- **Pretrained probes on the Hub.** Publish ready-to-use probes under the
  [`wrynx`](https://huggingface.co/wrynx) organization. Depends on probe
  save/load.
- **Streaming responses with probe signals.** Stream tokens and probe signals
  together, building on the reference SSE chunk format in
  `examples/openai_server/`.
- **vLLM pipeline parallelism.** Collect activations from workers that cover
  different layer ranges, and make inline (abort-capable) extraction points
  work across pipeline stages. **help wanted**
- **Published Docker image.** A container image with Undercurrent and a
  tested vLLM, so production users don't have to build their own.
- **Overhead reduction.** Measure and reduce the latency and throughput cost
  of capture and routing, especially for inline probes. **help wanted**

## Later

- **SGLang adapter.** Implement the `EngineAdapter` extension point
  ([`undercurrent.adapters.base`](src/undercurrent/adapters/base.py)) for
  SGLang. **help wanted**
- **TGI adapter.** Implement
  [`EngineAdapter`](src/undercurrent/adapters/base.py) for Hugging Face
  Text Generation Inference. **help wanted**
- **llama.cpp / llama-cpp-python adapter.** Implement
  [`EngineAdapter`](src/undercurrent/adapters/base.py) for llama.cpp,
  likely through llama-cpp-python. **help wanted**
- **Combined vLLM TP × PP.** Once tensor and pipeline parallelism each work
  on their own.
- **Wider vLLM coverage.** Support for vLLM's V2 model runner, `kv`
  extraction points, and negative-indexed position selectors such as
  `generated[-1]`, none of which the vLLM adapter supports today.
  **help wanted**
- **Activation-steering interventions.** Interventions beyond aborting
  generation, such as modifying activations in flight. Today a probe can
  `continue`, `flag` or `abort`.
- **GPU-side probes.** Run probes on the GPU next to the model, without
  copying activations to the host first.
- **Fluent Python spec builder.** A chainable Python API for building specs.
  Today, construct `ExtractionPoint` objects directly in Python, or write
  YAML.

Writing a new engine adapter? Start with the "New engine adapter" issue
template and the `EngineAdapter` checklist in
[CONTRIBUTING.md](CONTRIBUTING.md).

## How to influence the roadmap

- **Tell us what you need.** Open a thread in
  [Discussions](https://github.com/wrynx/undercurrent/discussions) describing
  your use case. Concrete use cases weigh more than feature requests.
- **Vote with reactions.** Add a 👍 to existing issues and Discussions you
  care about; maintainers use them when setting priorities.
- **Propose a change.** Open a feature request issue, or a pull request
  against this file explaining what should move and why. Changes are decided
  as described in [GOVERNANCE.md](GOVERNANCE.md).
- **Build it.** The fastest way to move an item up is to work on it. Items
  marked **help wanted** are especially welcome; please say you're working
  on one so others know.

Background on current limitations: [the vLLM adapter](docs/internals/vllm-adapter.md),
[tensor and pipeline parallelism](docs/internals/vllm-parallelism.md), and
[compatibility](docs/compatibility.md).

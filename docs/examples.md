# Examples

The [`examples/`](https://github.com/wrynx/undercurrent/tree/main/examples) directory
in the repository holds runnable examples. They are **not installed** with
`pip install undercurrent`; clone the repository to run them. Each example is
covered by tests under `tests/examples/` that run on CPU, so the code stays in
sync with the library.

## Content-safety demo

[`examples/content_safety/`](https://github.com/wrynx/undercurrent/tree/main/examples/content_safety)
shows both probe kinds wired end to end: a single-shot classifier head that
scores one activation, and a trajectory probe (a GRU cell) that keeps a running
score over the generated tokens and stops generation once it crosses a
threshold. It includes YAML specs that bind extraction points to the probes, a
script that makes a dummy MLP checkpoint, a vLLM pipeline script
(`run_vllm_pipeline.py`) and a Dockerfile for the reference server.

!!! warning "A wiring demo, not a content-safety product"
    The probes use random-initialised or dummy weights, so their scores mean
    nothing. The demo shows how probes plug into Undercurrent. Don't use it to
    filter content.

**Needs:** the tests run on CPU. The vLLM pipeline script needs a GPU and
vLLM installed.

## OpenAI-compatible server

[`examples/openai_server/`](https://github.com/wrynx/undercurrent/tree/main/examples/openai_server)
is a reference HTTP server that returns probe verdicts in OpenAI-style API
responses. `response_schema.py` defines the wire format (completion bodies, a
`probing` object and SSE chunks) and `serve_llama_mlp_pipeline.py` serves a
probed vLLM model over HTTP. It is a starting point to copy into your own
serving stack, not a supported `undercurrent` command, and its stdlib HTTP
server isn't hardened for production traffic.

**Needs:** the wire-format code runs on CPU. The server needs a GPU, vLLM and,
for the default Llama 3.1 8B model, access to that gated Hugging Face repo.

## Train a probe

[`examples/train_probe/`](https://github.com/wrynx/undercurrent/tree/main/examples/train_probe)
collects activations from a model with Undercurrent, trains a linear (or small
MLP) probe on them, reports accuracy, F1 and AUROC, and saves the probe with
`safetensors`. `use_probe.py` loads it back and runs it inline through
`ProbedModel`, aborting generation on flagged prompts. It ships a built-in toy
dataset and a loader for Civil Comments (CC0).

**Needs:** CPU is enough for the toy dataset with a small model such as GPT-2.
The Civil Comments run needs `datasets` (`examples/train_probe/requirements.txt`),
network access and, realistically, a GPU.

## Sample specs

[`examples/specs/`](https://github.com/wrynx/undercurrent/tree/main/examples/specs)
holds sample YAML specs, one per common pattern: a single-token probe on the
last prompt token, a trajectory over every second generated token run
asynchronously, an offset dual-position probe, and a sliced window of
generated tokens. Use them as templates for your own specs.

**Needs:** nothing beyond the base install; they are plain YAML.

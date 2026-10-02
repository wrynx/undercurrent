# Compatibility

This page lists the dependency versions Undercurrent supports, the combinations
that have actually been tested, and what happens when your vLLM version is
outside the supported range.

## Declared ranges

These are the ranges in `pyproject.toml`.

| Dependency     | Range              | Why |
|----------------|--------------------|-----|
| Python         | `>=3.10`           | |
| torch          | `>=2.1`, no ceiling | vLLM pins an exact torch version. A ceiling here would eventually conflict with that pin. |
| transformers   | `>=4.40,<6`        | The HF adapter's stopping criterion returns a per-sequence bool tensor, which has been the `StoppingCriteria` contract since 4.39. The adapter also hooks model structure (decoder-layer attribute paths, layer outputs) that can change in a major release. |
| pydantic       | `>=2,<3`           | The spec models use the pydantic v2 API. |
| PyYAML         | `>=6,<7`           | |
| numpy          | `>=1.23`, no ceiling | Not imported directly, so this never blocks a numpy that torch accepts. |
| packaging      | `>=22`             | Used by the runtime vLLM version check. |
| vllm (optional) | `>=0.28,<0.29`    | The adapter hooks undocumented vLLM internals, so the range stops at the next minor. Checked at runtime (see below). |

vLLM is never installed by default. There are two supported ways to use the
vLLM adapter:

1. **Install Undercurrent into your existing vLLM environment or image**
   (`pip install undercurrent` there). This is the recommended production
   path. pip doesn't check the vLLM range in this case, so the adapter checks
   it at runtime.
2. **`pip install "undercurrent[vllm]"`** installs a vLLM from the tested range.

CUDA is whatever your vLLM wheel or image provides. Undercurrent has no CUDA
requirement of its own.

## Tested combinations

Only combinations that have actually been run are listed here.

| Python | torch | transformers | vllm | CUDA | Status |
|--------|-------|--------------|------|------|--------|
| 3.14.4 | 2.13.0 (`+cu130` build) | 5.16.1 | 0.28.0 | As provided by the vLLM wheel; no GPU present | **Tested locally**: full unit test suite, CPU only. GPU and real vLLM generation tests were skipped. |
| 3.13.15 | 2.13.0 (`+cu130` build) | 5.17.0 | 0.28.0 | 13.0, NVIDIA L4 (driver 580.82.07) | **GPU-verified, 2026-10-02**: `scripts/gpu_check.sh` passed, covering every GPU test (vLLM integration, `ProbedModel` with the vLLM backend, and the `residual_stream` HF-vs-vLLM equivalence check) and the vLLM adapter suite against the real vLLM. Run on Google Colab. |
| 3.10.21 | 2.1.2 (`+cpu` build) | 4.40.2 | not installed | n/a | **The declared lower bounds** (with pydantic 2.0.3, PyYAML 6.0.3, numpy 1.23.5, packaging 22.0): CPU unit test suite passes locally. The CI `floors` job installs the same versions and runs the suite. Tests tied to newer releases skip themselves (the byte-for-byte JSON Schema comparisons need pydantic 2.11+; the content-safety demo needs torch 2.6+ to load checkpoints). |

There is no GPU CI for v0.1. Instead, before every release a maintainer runs
`scripts/gpu_check.sh` on a GPU machine with vLLM installed. It runs the GPU
and real-vLLM tests and prints the combination it ran with, which is then
added to the table above (see [RELEASING.md](https://github.com/wrynx/undercurrent/blob/main/RELEASING.md)).
A vLLM row without such a GPU run is "declared, not yet GPU-verified". You
can run the same check on your own GPU machine:

```bash
pip install -e ".[dev]"     # from a checkout, into your vLLM environment
scripts/gpu_check.sh
```

## Known issues (v0.1)

These affect the vLLM adapter only. The Hugging Face backend is not affected.

### Fixed in 0.1.0: `residual_stream` on fused-residual vLLM layers

**Fixed in 0.1.0: `residual_stream` on fused-residual layers is captured as
`hidden_states + residual`. GPU-verified on 2026-10-02 (NVIDIA L4, vLLM 0.28.0,
torch 2.13.0+cu130) with `scripts/gpu_check.sh`.**

- **The problem.** `residual_stream` is a forward hook on the decoder layer.
  vLLM's fused-residual decoder layers (for example `LlamaDecoderLayer.forward`
  in `vllm/model_executor/models/llama.py`, vLLM 0.28) don't add the residual
  themselves. They return `(hidden_states, residual)`, where `hidden_states`
  is the MLP output (the post-feedforward-normed MLP output on Gemma 2 and 3)
  and `residual` is the stream before that contribution is added. The add
  happens in the next layer's fused `input_layernorm`, or in the final norm.
  Before the fix the adapter captured only the first element, so on these
  models `residual_stream` held the MLP output, not the residual stream.
- **The fix.** For a `(hidden_states, residual)` pair of tensors with the
  same shape and dtype, the adapter now captures their sum: the residual
  stream after the layer, which is what the HF adapter captures and what vLLM
  itself computes when it needs the stream (EAGLE's auxiliary hidden states).
  The sum is computed in the model's dtype, bit-identical to vLLM's own fused
  add, and is a new tensor, so the next layer's in-place fused kernel can't
  change it. A layer that returns a single tensor (GPT-2, Granite) or
  `(hidden_states, None)` (MiniCPM, FlexOlmo, Gemma 4) is captured as before.
  Any other output (for example HunYuan's 3-tuple) raises
  `VLLMAdapterLimitationError` when a `residual_stream` extraction point
  needs it, rather than capturing a guess.
- **Affected models.** Every vLLM architecture whose decoder layer returns
  `(hidden_states, residual)`. In vLLM 0.28 that's 73 model files, including
  Llama (and Llama-derived models such as Phi-3), Mistral, Mixtral, Qwen2,
  Qwen2-MoE, Qwen3, Qwen3-MoE, Gemma, Gemma 2, Gemma 3, DeepSeek-V2/V3,
  Command R and OLMoE.
- **Tensor and pipeline parallelism.** The fix holds under both; see
  [Tensor & pipeline parallelism](internals/vllm-parallelism.md#residual_stream).
- **GPU verification.** `tests/adapters/vllm/test_residual_stream_gpu.py`
  loads a tiny random Llama and GPT-2 with both backends and checks that the
  per-token `residual_stream` captures match (relative error at most 1e-3).
  It passed on an NVIDIA L4 with vLLM 0.28.0 (see
  [Tested combinations](#tested-combinations)). CPU tests check the
  capture against fake layers with vLLM's contract, and an upstream-contract
  test fails if the installed vLLM's `LlamaDecoderLayer` stops returning
  `(hidden_states, residual)`.
- **Records from earlier builds.** `residual_stream` records captured on
  vLLM before this fix, from these models, hold the MLP output. Re-capture
  them, or use the HF backend.

### No GPU CI for the vLLM adapter

There is no GPU CI for v0.1. The GPU and real-vLLM tests run by hand: a
maintainer runs
[`scripts/gpu_check.sh`](https://github.com/wrynx/undercurrent/blob/main/scripts/gpu_check.sh)
on a GPU machine before each release, and the combination it ran with goes
into [Tested combinations](#tested-combinations). For 0.1.0 that's an NVIDIA
L4 with vLLM 0.28.0. Other GPUs, larger models and multi-GPU topologies
haven't been run, and the vLLM adapter is still experimental. You can run the
check on your own GPU machine too (see above).

### vLLM 0.29 and 0.30 aren't supported yet

The supported range is `vllm>=0.28,<0.29`. vLLM 0.30 is already out, so a
current vLLM environment will probably fail the
[runtime version check](#the-runtime-vllm-version-check) with
`VLLMAdapterLimitationError`. Set `UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM=1` to
turn the error into a warning and try it anyway: the adapter's introspection
checks still fail loudly if vLLM's internals have changed, but a run that
doesn't fail isn't validated. The supported range only widens to a newer
minor after `scripts/gpu_check.sh --allow-unsupported-vllm` passes against it
on a GPU (see the [support policy](#support-policy)).

## Support policy

- **vLLM:** we support the latest vLLM minor that we've validated. Older
  minors are best-effort for one Undercurrent release after support for a new
  minor lands, then they're dropped. The supported range only widens after
  `scripts/gpu_check.sh` passes against the new minor (run it with
  `--allow-unsupported-vllm` before the range is widened).
- **torch:** any version from 2.1 that your vLLM or CUDA stack requires.
- **transformers:** the declared range. A new major (6.x) is supported only
  after it has been tested.
- **Python:** 3.10 and newer.

## The runtime vLLM version check

Constructing a `VLLMEngineAdapter`, or vLLM initialising the
`ProbingWorkerExtension`, checks the installed vLLM version against
`vllm>=0.28,<0.29` (`SUPPORTED_VLLM` in
`undercurrent/adapters/vllm/version_check.py`). The check reads package
metadata only. It does not import vLLM, and importing
`undercurrent.adapters.vllm` doesn't run it.

Outside the range it raises `VLLMAdapterLimitationError`. The error names the
installed version, the supported range, and the override:

```text
undercurrent.adapters.vllm supports vllm>=0.28,<0.29, but vllm==0.29.0 is installed. ...
```

### Overriding the check

To try an untested vLLM version anyway, for example while evaluating a new
release, set:

```bash
export UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM=1
```

The error then becomes a `RuntimeWarning`. Activation capture may still fail,
loudly, if vLLM's internals have changed: the adapter's introspection layer
validates the model-runner layout it reads and raises `VLLMIntrospectionError`
rather than mis-capturing. Please report what you find on
[GitHub issues](https://github.com/wrynx/undercurrent/issues).

If vLLM isn't installed at all, `load_model()` raises `MissingDependencyError`,
which lists the same two install routes.

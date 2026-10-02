# probing_adapter_hf

HuggingFace `transformers` engine adapter for the activation-probing
platform, driven through a single sequential `model.generate()` call per
request. Now `undercurrent.adapters.hf` (`src/undercurrent/adapters/hf/`).
The `EngineAdapter` contract every adapter implements lives in
`src/undercurrent/adapters/base.py` (`undercurrent.adapters.base`); this
package re-exports it.

## Why this adapter has the cleanest intervention semantics

Every other adapter has to reconcile mid-generation intervention
(`InterventionPolicy.mode="block_until_signal"`) with however its engine
batches/schedules work. This adapter doesn't: `generate()` drives exactly
one sequential decode loop for exactly one request at a time (see
`HFEngineAdapter`'s docstring for what that rules out -- no concurrent
`generate()` calls, no beam search / batch size > 1). "The decode step" IS
the whole engine for that one request, so a bounded wait for a probe's
decision is scoped to exactly what the caller asked for, with no
collateral stalling of unrelated requests -- unlike
`undercurrent.adapters.vllm`, where the same wait can widen to every request
sharing a continuously-batched scheduler step (see that package's
`worker_extension.py`, "INTERVENTION TIMEOUT LIMITATIONS").

See `src/undercurrent/adapters/hf/stopping_criteria.py`'s module docstring for
exactly how `InterventionPolicy.timeout_ms`/`on_timeout` end up enforced
here: `Router.route()` itself already bounds the wait; the
`StoppingCriteria` just reads the flag a forward hook already set.

## Interception strategy

`torch.nn.Module.register_forward_hook` on the model's decoder layer /
attention / MLP submodules, located via a short list of plausible
attribute paths (`model.transformer.h` for GPT-2-style models,
`model.model.layers` for Llama-family models). `model.forward` itself is
also wrapped, purely to count decode steps -- see `_ActiveGeneration`'s
docstring in `adapter.py` for why a call-counter (not tensor shape) is
what actually distinguishes the prefill call from a decode-step call (a
1-token prompt's prefill call also has seq_len==1, indistinguishable from
a decode step by shape alone).

## Known limitations (documented, not silent gaps)

- One `generate()` call at a time -- no batching/event-loop layer here to
  interleave concurrent requests, unlike `undercurrent.adapters.vllm`.
- No beam search / no batch size > 1 -- position tracking assumes exactly
  one sequence advancing one token at a time.
- `tensor_type="kv"` extraction points are not supported, for the same
  reason `undercurrent.adapters.vllm` doesn't support them.

## Usage

```python
from undercurrent.adapters.hf import HFEngineAdapter
from undercurrent.router import Router

adapter = HFEngineAdapter()
adapter.load_model("gpt2")  # or an already-constructed model + tokenizer= kwarg

router = Router(probe_registry={...})
adapter.register_extraction(request_id, extraction_points)
text = adapter.generate(request_id, prompt, {"max_new_tokens": 32}, router)
adapter.unregister_extraction(request_id)
```

torch and transformers are base dependencies of `undercurrent`, so a plain
`pip install -e .` at the repo root is enough for real generation. The
`EngineAdapter` ABC itself is pure Python and imports neither.

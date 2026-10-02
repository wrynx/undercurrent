# Content-safety demo

**This is a demo, not a safety product.** The probes wrap small PyTorch models
(a classifier head and a GRU-cell recurrence) with **random-initialised or
dummy weights**. Their scores mean nothing. The demo is here to show how probes
plug into Undercurrent end to end. It is **not installed** with
`pip install undercurrent`, and there is no extra for it.

## What it demonstrates

- **Both probe kinds**: `SingleTokenSafetyProbe` (`single_shot`) scores one
  activation, and `TrajectorySafetyProbe` (`trajectory`) keeps a running score
  over generated tokens and aborts generation once it crosses a threshold.
  `CustomMLPProbe` loads a checkpoint you supply.
- **Spec binding**: `content_safety_demo/spec_binding.py` maps each YAML
  extraction point (`content_safety_*.yaml`, `custom_mlp_llama31_8b.yaml`) to a
  probe class and its constructor kwargs.
- **The vLLM pipeline**: `run_vllm_pipeline.py` runs spec → `Router` → live vLLM
  generation → probe verdicts. `../openai_server/serve_llama_mlp_pipeline.py`
  serves the same wiring over HTTP. `Dockerfile` packages that server.

## Running it

Run these from the repo root. They need a GPU and `vllm` installed:

```bash
pip install -e .    # plus the vLLM adapter
python examples/content_safety/run_vllm_pipeline.py --model gpt2 --prompt "Hello" --max-tokens 32

python examples/content_safety/make_dummy_mlp_checkpoint.py --input-dim 4096 --out /tmp/mlp_probe.pt
PROBE_MODEL_PATH=/tmp/mlp_probe.pt python examples/openai_server/serve_llama_mlp_pipeline.py --port 8000
```

Scripts in this directory import the package `content_safety_demo` from their
own directory. The tests live in `tests/examples/content_safety/` and run
without a GPU: `python -m pytest tests/examples/content_safety`.

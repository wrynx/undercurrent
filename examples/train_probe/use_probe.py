"""Run a trained probe inline. Usage: python examples/train_probe/use_probe.py [PROBE_DIR] [MODEL]"""

import sys

from linear_probe import PROBE_TYPE, load_probe, read_meta, spec_for

from undercurrent import ProbedModel

probe_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/train_probe"
factory = load_probe(probe_dir)  # safetensors + probe.json -> ProbeFactory
meta = read_meta(probe_dir)  # where to read: layer, tensor_type, position
model = sys.argv[2] if len(sys.argv) > 2 else meta["base_model"]

prompts = ["Thanks for sharing, the recipe looks great.", "Shut up, you are a worthless idiot."]
with ProbedModel.from_pretrained(model, spec=spec_for(meta), probes={PROBE_TYPE: factory}) as m:
    for out in m.generate(prompts, max_new_tokens=20, temperature=0.0):
        verdict = out.probe_results["trained_probe"].verdict
        status = f"ABORTED ({out.abort_reason})" if out.aborted else f"ok: {out.text!r}"
        print(f"{out.prompt!r}\n  score={verdict['score']:.3f} label={verdict['label']} -> {status}")

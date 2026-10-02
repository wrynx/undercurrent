# Undercurrent quickstart: record the residual-stream norm at the last prompt token. Usage: python examples/quickstart.py [MODEL=gpt2]
import sys

from undercurrent import Probe, ProbedModel, ProbeResult, register_probe


@register_probe("norm")
class NormProbe(Probe):
    probe_kind = "single_shot"

    def on_start(self, ctx):
        self.norm = None

    def on_activation(self, record):
        self.norm = round(float(record.tensor.norm()), 2)

    def on_end(self, ctx):
        return ProbeResult(ctx.request_id, self.extraction_point_name, verdict=self.norm)


SPEC = "extraction_points: [{name: prompt_norm, layers: 0, tensor_type: residual_stream, position: 'prompt[-1]', probe_type: norm, probe_kind: single_shot}]"
with ProbedModel.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else "gpt2", spec=SPEC) as model:
    out = model.generate("The weather today is", max_new_tokens=20, temperature=0.0)
print(out)  # GenerationOutput(text=..., probe_results={prompt_norm: verdict=...})

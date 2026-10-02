"""A stateless probe written as a plain function. Run: python examples/function_probe.py"""

import torch

from undercurrent import ActivationRecord, probe

head = torch.nn.Linear(8, 1)  # stand-in for a trained probe head


@probe("toxicity", threshold=0.8)  # registered by name: use probe_type: toxicity in a spec
def toxicity(record) -> float:
    return head(record.tensor).sigmoid().item()


record = ActivationRecord("req-1", "ep-1", 6, 0, "residual_stream", tensor=torch.randn(8), is_generated=True)
print("score:", toxicity.fn(record))  # the original function, handy in unit tests
signal = toxicity.spawn("req-1", "ep-1").on_activation(record)
print("action:", signal.action.value, "metadata:", signal.metadata)

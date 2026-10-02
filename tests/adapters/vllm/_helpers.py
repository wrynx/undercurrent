"""Test helpers shared by the vLLM adapter tests (not fixtures -- those live in
conftest.py). Import them absolutely: `from tests.adapters.vllm._helpers import X`.

`make_extraction_point` builds an `ExtractionPoint` with test-friendly
defaults. The `Fake*` classes stand in for vLLM's model runner / torch modules
for hook-installation bookkeeping only (see test_worker_extension_bookkeeping.py's
module docstring) -- no vLLM or torch needed.
"""

from collections.abc import Callable
from typing import Any

from undercurrent.spec import ExecutionMode, ExtractionPoint, ProbeKind, TensorType, parse_position


def make_extraction_point(
    name="ep-1",
    layer=0,
    tensor=TensorType.RESIDUAL_STREAM,
    position="generated[*]",
    probe_type="recording",
    probe_kind=ProbeKind.TRAJECTORY,
    execution_mode=ExecutionMode.INLINE,
    stride=None,
    until=None,
    queue_depth=None,
    intervention=None,
) -> ExtractionPoint:
    layers = (layer,) if isinstance(layer, int) else tuple(layer)
    return ExtractionPoint(
        name=name,
        layers=layers,
        tensor_type=tensor,
        position=parse_position(position),
        stride=stride,
        until=until,
        probe_type=probe_type,
        probe_kind=probe_kind,
        execution_mode=execution_mode,
        queue_depth=queue_depth,
        intervention=intervention,
    )


class FakeHookHandle:
    def __init__(self, owner: "FakeModule", fn: Callable) -> None:
        self._owner = owner
        self._fn = fn

    def remove(self) -> None:
        self._owner.hooks.remove(self._fn)


class FakeModule:
    """Stands in for a torch.nn.Module for hook-installation purposes only
    (register_forward_hook bookkeeping) -- never actually invoked/forwarded
    through in these tests, so it never needs to produce a real tensor."""

    def __init__(self, self_attn: "FakeModule" = None, mlp: "FakeModule" = None) -> None:
        self.hooks: list[Callable] = []
        self.self_attn = self_attn
        self.mlp = mlp

    def register_forward_hook(self, fn: Callable) -> FakeHookHandle:
        self.hooks.append(fn)
        return FakeHookHandle(self, fn)


class FakeInnerModel:
    def __init__(self, layers: list[FakeModule]) -> None:
        self.layers = layers


class FakeModel:
    def __init__(self, layers: list[FakeModule]) -> None:
        self.model = FakeInnerModel(layers)


class FakeModelRunner:
    def __init__(self, num_layers: int = 2) -> None:
        self.model = FakeModel([FakeModule(self_attn=FakeModule(), mlp=FakeModule()) for _ in range(num_layers)])
        self.execute_model_calls = 0

        def _execute_model(scheduler_output: Any, *a: Any, **kw: Any) -> str:
            self.execute_model_calls += 1
            return "ok"

        self.execute_model = _execute_model

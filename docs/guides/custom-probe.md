# Write a custom probe

<!-- owner: p3-custom-probe-guide -->

This guide takes you from nothing to a probe that is registered, tested,
running on a real model, and packaged so that anyone can `pip install` it.
Along the way you build the same idea three ways: a threshold on the norm of
the residual stream, first as a plain function, then as a class, then as a
stateful trajectory probe that stops generation when a running average gets
too high.

All the Python on this page runs on a CPU, top to bottom, and later blocks use
names defined in earlier ones. Paste them into one script or one notebook in
order. The only model used is GPT-2, in [section 7](#7-run-it-on-a-real-model).

!!! tip "Background"
    [Probes & lifecycle](../concepts/probes.md) explains what a probe is and
    when its methods are called. [Extraction points & specs](../concepts/extraction-points.md)
    explains how a spec decides which activations reach which probe. You can
    follow this guide without reading them first.

## 1. Start with a function probe

Most probes answer one question about one activation: "how large is this?",
"how toxic is this?". They keep no state between activations. For those, a
plain function is enough. Decorate it with `@probe(...)`:

```python
import torch

from undercurrent.core import ActivationRecord
from undercurrent.core.function_probe import probe


@probe("norm_threshold", threshold=150.0)
def norm_threshold(record: ActivationRecord) -> float:
    """L2 norm of one activation vector."""
    return float(record.tensor.float().norm())
```

The function takes one `ActivationRecord` and returns a score. The record
carries the captured tensor (`record.tensor`, a 1-D `torch.Tensor` of size
`hidden_size` with the HF adapter) and where it came from:
`record.layer`, `record.token_pos`, `record.tensor_type`,
`record.is_generated`, `record.request_id` and `record.extraction_point_name`.

`@probe(...)` does two things:

1. It turns the function into a real `Probe` subclass with
   `probe_kind = "single_shot"`. `norm_threshold` is now a class, and the
   original function stays available as `norm_threshold.fn`.
2. It registers that class under the name `"norm_threshold"`, so a spec can
   refer to it with `probe_type: norm_threshold`.

The full signature is:

```py
probe(name=None, /, *, threshold=None, action="abort", registry=None, register=True)
```

- `name`: the `probe_type` to register under. Defaults to the function's
  `__name__`, and bare `@probe` without parentheses also works.
- `threshold`: a numeric score `>= threshold` flags the activation. With
  `None` (the default), scores are recorded but never flag anything.
- `action`: what a flagged activation asks for. `"abort"` (the default) stops
  generation. `"flag"` records the hit and lets generation continue.
- `registry`: a `ProbeRegistry` to register in, instead of the global one.
- `register=False`: create the class without registering it anywhere.

Try it on a synthetic activation. A helper that builds a 768-dimensional vector
with a chosen norm makes the numbers easy to follow, and the rest of the page
reuses it:

```python
HIDDEN = 768  # GPT-2's hidden size


def vec(norm: float) -> torch.Tensor:
    """A HIDDEN-sized vector whose L2 norm is exactly `norm`."""
    v = torch.zeros(HIDDEN)
    v[0] = norm
    return v


record = ActivationRecord(
    request_id="req-1",
    extraction_point_name="gate",
    layer=6,
    token_pos=4,
    tensor_type="residual_stream",
    tensor=vec(200.0),
    is_generated=True,
)

print(norm_threshold.fn(record))  # the plain function: 200.0

instance = norm_threshold.spawn("req-1", "gate")
signal = instance.on_activation(record)
print(signal.action, signal.confidence, signal.metadata)
# ProbeAction.ABORT 200.0 {'score': 200.0, 'probe': 'norm_threshold'}
assert signal.action.value == "abort"
```

### What the return value means

Each call returns one of four things. The decorator turns it into a
`ProbeSignal` (or nothing) for that activation:

| Return | Meaning | Signal emitted |
| --- | --- | --- |
| `float` / `int` | A score. | `abort` (or `flag`) if `threshold` is set and `score >= threshold`, otherwise `continue`. `confidence` is the score. |
| `bool` | A yes/no decision. | `True` → `abort` (or `flag`), with `confidence=1.0`. `False` → `continue`. |
| `ProbeSignal` | Full control. | Passed through untouched. |
| `None` | Nothing to say about this activation. | None. The activation isn't counted. |

Every non-`None` return produces a signal, including `continue`, with metadata
`{"score": ..., "probe": "<name>"}`. That is how scores reach the signal
history and, for async points, the sinks. When the request ends, the probe's
`ProbeResult.verdict` is `{"flagged": bool, "max_score": float | None, "n": int}`.

Here are the other return types in action. `finite_guard` returns `None` for
prompt tokens and a `bool` otherwise. `norm_report` returns a score while all
is well and builds its own `ProbeSignal` when it wants to attach extra
metadata. It also takes an option: extra parameters must be keyword-only, and
their values come from the spec's `probe_args` (see
[section 5](#5-register-it)).

```python
from undercurrent.core import ProbeAction, ProbeSignal


@probe("finite_guard")
def finite_guard(record: ActivationRecord) -> bool | None:
    if not record.is_generated:
        return None  # ignore prompt tokens entirely
    return not bool(torch.isfinite(record.tensor).all())  # True -> abort


@probe("norm_report")
def norm_report(record: ActivationRecord, *, limit: float = 100.0) -> float | ProbeSignal:
    norm = float(record.tensor.float().norm())
    if norm < limit:
        return norm  # no threshold set, so this is recorded, never flagged
    return ProbeSignal(
        action=ProbeAction.FLAG,
        confidence=norm / limit,
        metadata={"norm": norm, "limit": limit, "layer": record.layer},
    )


nan_record = ActivationRecord("req-1", "guard", 6, 4, "residual_stream", vec(1.0) * float("nan"), is_generated=True)
assert finite_guard.spawn("req-1", "guard").on_activation(nan_record).action is ProbeAction.ABORT

reporter = norm_report.spawn("req-1", "report", limit=150.0)
print(reporter.on_activation(record).metadata)  # {'norm': 200.0, 'limit': 150.0, 'layer': 6}
```

### When you outgrow a function

A function probe is stateless and `single_shot`. Write a class (next sections)
when you need any of these:

- **Per-request state**: a running score, a counter, a history of earlier
  activations.
- **A trajectory**: a decision that depends on many activations, in order.
- **Setup or teardown**: work in `on_start` (read the prompt metadata, allocate
  buffers) or a custom verdict built in `on_end`.

## 2. Pick a probe kind

Every probe class declares a `probe_kind`, and every extraction point in a spec
declares the `probe_kind` it expects. The router refuses to bind a probe to a
point whose kind doesn't match.

| | `single_shot` | `trajectory` |
| --- | --- | --- |
| Question it answers | "Is this activation bad?" | "Is this generation going somewhere bad?" |
| State between activations | None needed | Yes: running scores, windows, counters |
| Typical position | One token: `prompt[-1]`, `generated[0]` | Many tokens: `generated[*]`, `generated[5:]` |
| `execution_mode: inline` (can stop generation) | Yes | Yes |
| `execution_mode: async` (observe only, off the hot path) | **Not allowed** | Yes |
| Function probe (`@probe`) | Yes | No, write a class |

A `single_shot` probe still receives every activation its point matches. The
kind describes the probe's intent, and it decides whether async execution is
allowed: async only makes sense for a probe that keeps state across calls. See
[Execution modes: inline vs async](../concepts/execution-modes.md) for the
trade-off.

## 3. Write a single-shot class probe

Here is the same threshold as a `Probe` subclass. A class has three methods to
implement, called in this order for each request:

1. `on_start(request_ctx)`: once, before any activation. Set up per-request
   state here.
2. `on_activation(record)`: once per matching activation. Return a
   `ProbeSignal` to act on this activation, or `None`.
3. `on_end(request_ctx)`: exactly once, when generation finishes or is
   aborted. It **must always return a `ProbeResult`**, even if
   `on_activation` was never called (the point never matched a token, or
   generation was aborted by another probe before reaching it).

Constructor arguments are the probe's options. They come from
`ProbeFactory` kwargs and from the spec's `probe_args`.

```python
from undercurrent.core import Probe, ProbeResult, RequestContext
from undercurrent.core.registry import register_probe


@register_probe("norm_threshold_class")
class NormThresholdProbe(Probe):
    """Aborts on the first activation whose L2 norm reaches `threshold`.

    Verdict: {"flagged": bool, "max_norm": float | None, "n": int}
    """

    probe_kind = "single_shot"

    def __init__(self, threshold: float = 150.0) -> None:
        super().__init__()  # required: sets up request_id / extraction_point_name
        self.threshold = threshold

    def on_start(self, request_ctx: RequestContext) -> None:
        # Per-request state lives on `self`. Each request gets a fresh instance.
        self.max_norm: float | None = None
        self.n = 0
        self.signals: list[ProbeSignal] = []

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        norm = float(record.tensor.float().norm())
        self.n += 1
        self.max_norm = norm if self.max_norm is None else max(self.max_norm, norm)
        if norm < self.threshold:
            return None
        signal = ProbeSignal(
            action=ProbeAction.ABORT,
            confidence=norm,
            metadata={"norm": norm, "threshold": self.threshold, "token_pos": record.token_pos},
        )
        self.signals.append(signal)
        return signal

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"flagged": bool(self.signals), "max_norm": self.max_norm, "n": self.n},
            signal_history=list(self.signals),
            metadata={"threshold": self.threshold},
        )
```

`@register_probe("norm_threshold_class")` registers the class by name; more on
that in [section 5](#5-register-it).

You never call `NormThresholdProbe(...)` yourself in real code. The router
creates one instance per `(request, extraction point)` with
`NormThresholdProbe.spawn(request_id, extraction_point_name, **kwargs)` and
throws it away when the request ends. You can drive that lifecycle by hand to
check the probe, including the case where no activation ever arrives:

```python
ctx = RequestContext(request_id="req-1", prompt_metadata={}, extraction_point_config=None)

p = NormThresholdProbe.spawn("req-1", "gate", threshold=150.0)
p.on_start(ctx)
assert p.on_activation(ActivationRecord("req-1", "gate", 6, 3, "residual_stream", vec(90.0), True)) is None
abort = p.on_activation(ActivationRecord("req-1", "gate", 6, 4, "residual_stream", vec(160.0), True))
assert abort.action is ProbeAction.ABORT
result = p.on_end(ctx)
print(result.verdict)  # {'flagged': True, 'max_norm': 160.0, 'n': 2}

# A probe that never saw an activation still returns a result.
idle = NormThresholdProbe.spawn("req-2", "gate")
idle_ctx = RequestContext("req-2", {}, None)
idle.on_start(idle_ctx)
assert idle.on_end(idle_ctx).verdict == {"flagged": False, "max_norm": None, "n": 0}
```

## 4. Write a trajectory probe

A trajectory probe makes its decision from the whole sequence of activations
it has seen so far. This one keeps an exponential moving average (EMA) of the
activation norm and aborts the first time the average crosses a threshold. A
single spike barely moves the average; a sustained rise does.

```python
@register_probe("norm_ema")
class NormEMAProbe(Probe):
    """Exponential moving average of the activation norm; aborts once it reaches `threshold`.

    Verdict: {"ema": float | None, "count": int, "aborted": bool}
    """

    probe_kind = "trajectory"

    def __init__(self, threshold: float = 120.0, alpha: float = 0.3) -> None:
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.threshold = threshold
        self.alpha = alpha

    def on_start(self, request_ctx: RequestContext) -> None:
        self.ema: float | None = None
        self.count = 0
        self.aborted = False
        self.signals: list[ProbeSignal] = []

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        norm = float(record.tensor.float().norm())
        self.ema = norm if self.ema is None else self.alpha * norm + (1 - self.alpha) * self.ema
        self.count += 1
        if self.aborted or self.ema < self.threshold:
            return None  # abort once, then stay quiet
        self.aborted = True
        signal = ProbeSignal(
            action=ProbeAction.ABORT,
            confidence=self.ema,
            metadata={"ema": self.ema, "count": self.count, "token_pos": record.token_pos},
        )
        self.signals.append(signal)
        return signal

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"ema": self.ema, "count": self.count, "aborted": self.aborted},
            signal_history=list(self.signals),
            metadata={"threshold": self.threshold, "alpha": self.alpha},
        )
```

Two details matter for trajectory probes:

- **Abort once.** After the abort, generation stops, but activations already
  in flight (and, at async points, everything still queued) can still reach
  `on_activation`. Returning `None` after the first abort keeps the signal
  history clean.
- **The verdict is final only in `on_end`.** At an async point, `on_end` runs
  after the router has drained the probe's queue, so the verdict reflects
  every activation it accepted.

## 5. Register it

The router finds a probe by the `probe_type` string in the spec. There are
three ways to make a name resolvable.

**The global registry.** `@probe("name")` and `@register_probe("name")`
(used above) both add to the process-wide registry,
`undercurrent.core.registry.default_registry`. A `Router()` built without a
registry argument looks names up there, and also in installed plugins
(see [section 9](#9-ship-it-as-a-plugin)). `register_probe` also has a
functional form, which is handy for registering a preconfigured variant:

```python
from undercurrent.core.registry import get_probe_factory, list_probes

# The same class, under a second name, with different default kwargs.
register_probe("norm_ema_strict", NormEMAProbe, threshold=90.0, alpha=0.5)

print([name for name in list_probes() if name.startswith(("norm", "finite"))])
# ['finite_guard', 'norm_ema', 'norm_ema_strict', 'norm_report', 'norm_threshold', 'norm_threshold_class']
print(get_probe_factory("norm_ema_strict").probe_kwargs)  # {'threshold': 90.0, 'alpha': 0.5}
```

Registering a name that is already taken raises `ValueError` (pass
`override=True` to replace it on purpose). Re-running the same class
definition, for example a notebook cell, is allowed. A misspelled
`probe_type` raises `ProbeNotFoundError`, which lists the known names and the
closest matches.

**An explicit registry.** Pass `Router` a dict of `probe_type` → `Probe`
subclass or `ProbeFactory`. The dict is authoritative: no fallback to the
global registry or to plugins. This is the most predictable choice for tests
and for services that should run exactly the probes they list.
`ProbeFactory(cls, kwargs)` binds default constructor kwargs:

```python
from undercurrent.core import ProbeFactory
from undercurrent.router import Router

explicit_router = Router(
    {
        "norm_threshold": norm_threshold,  # a function probe is a Probe subclass too
        "norm_ema": ProbeFactory(NormEMAProbe, {"threshold": 120.0, "alpha": 0.3}),
    }
)
explicit_router.shutdown()
```

You can also build a `ProbeRegistry()` of your own, and pass it to
`@probe(..., registry=...)`, `registry.register(...)` and `Router(registry)`.

**The spec.** Now refer to the probes by name. This spec puts both probes on
GPT-2's residual stream at layer 6, for every generated token:

```yaml
# probes.yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/wrynx/undercurrent/main/schema/probe-spec.schema.json
version: "1"
extraction_points:
  - name: gate
    layers: 6
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: norm_threshold
    probe_kind: single_shot
    execution_mode: inline
    probe_args:
      threshold: 150.0
  - name: drift
    layers: 6
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: norm_ema
    probe_kind: trajectory
    execution_mode: inline
    probe_args:
      threshold: 120.0
      alpha: 0.3
```

`probe_args` are passed to the probe's constructor (for a function probe:
`threshold`, `action` and the function's keyword-only parameters). They are
merged over the `ProbeFactory` kwargs, and the spec wins. Load the spec from a
file with `load_yaml_file("probes.yaml")`, or from a string as here:

```python
from undercurrent.spec import parse_yaml

SPEC_YAML = """
version: "1"
extraction_points:
  - name: gate
    layers: 6
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: norm_threshold
    probe_kind: single_shot
    execution_mode: inline
    probe_args:
      threshold: 150.0
  - name: drift
    layers: 6
    tensor_type: residual_stream
    position: "generated[*]"
    probe_type: norm_ema
    probe_kind: trajectory
    execution_mode: inline
    probe_args:
      threshold: 120.0
      alpha: 0.3
"""

spec = parse_yaml(SPEC_YAML)
print(spec.names)  # ('gate', 'drift')
```

To check a spec file and its probe names from the command line, import the
module that registers your probes:

```bash
undercurrent validate probes.yaml --check-probes --import my_package.probes
```

## 6. Test it without a model

The router doesn't care where activations come from. Feed it synthetic
`ActivationRecord`s and you can test the whole path (spec → registry → spawn
→ `on_activation` → signal → `on_end` → `ProbeResult`) in milliseconds,
without loading a model. These are ordinary pytest tests; in a test file, drop
the calls at the bottom and let pytest collect them.

```python
def make_record(request_id: str, point_name: str, gen_index: int, norm: float) -> ActivationRecord:
    """An activation as the adapter would emit it for a 5-token prompt."""
    return ActivationRecord(
        request_id=request_id,
        extraction_point_name=point_name,
        layer=6,
        token_pos=5 + gen_index,
        tensor_type="residual_stream",
        tensor=vec(norm),
        is_generated=True,
    )


def drive(router: Router, norms: list[float]):
    """Route one activation per generated token to every point; stop at the first abort."""
    abort = None
    with router.request(spec, request_id="test-req") as req:
        for i, norm in enumerate(norms):
            for point in spec:
                signal = req.route(make_record(req.request_id, point.name, i, norm))
                if signal is not None and signal.action is ProbeAction.ABORT:
                    abort = (point.name, signal)
                    break
            if abort:
                break
    return abort, req.results  # results are available after the `with` block


def test_quiet_generation_runs_to_the_end():
    with Router({"norm_threshold": norm_threshold, "norm_ema": NormEMAProbe}) as router:
        abort, results = drive(router, [90.0] * 10)
    assert abort is None
    assert results["gate"].verdict == {"flagged": False, "max_score": 90.0, "n": 10}
    assert results["drift"].verdict["aborted"] is False
    assert results["drift"].verdict["count"] == 10


def test_sustained_rise_aborts_on_the_trajectory():
    with Router({"norm_threshold": norm_threshold, "norm_ema": NormEMAProbe}) as router:
        abort, results = drive(router, [90.0, 90.0, 130.0, 130.0, 130.0, 130.0, 130.0, 90.0])
    point, signal = abort
    assert point == "drift"
    assert signal.metadata["count"] == 6  # EMA crosses 120 on the 6th token
    drift = results["drift"]
    assert drift.verdict["aborted"] is True
    assert drift.signal_history == [signal]
    assert results["gate"].verdict["flagged"] is False  # no single spike reached 150


def test_single_spike_aborts_on_the_gate():
    with Router({"norm_threshold": norm_threshold, "norm_ema": NormEMAProbe}) as router:
        abort, results = drive(router, [90.0, 90.0, 400.0, 90.0])
    point, signal = abort
    assert point == "gate"
    assert signal.metadata == {"score": 400.0, "probe": "norm_threshold"}
    assert results["gate"].verdict["flagged"] is True
    assert results["gate"].verdict["n"] == 3  # nothing routed after the abort


test_quiet_generation_runs_to_the_end()
test_sustained_rise_aborts_on_the_trajectory()
test_single_spike_aborts_on_the_gate()
print("all probe tests passed")
```

A few habits that keep probe tests reliable:

- Use the router as a context manager (`with Router(...) as router:`), so its
  worker threads shut down at the end of each test.
- Read `req.results` **after** the `with router.request(...)` block. That is
  when `end_request` has run and every `on_end` has returned.
- For async points, the probe runs on a background thread. Assert on
  `req.results` (complete once the request has ended), not on signals returned
  by `route()`, which is always `None` for async points.
- Pass a dict registry, as above, so the test doesn't depend on what else
  happens to be registered globally.

## 7. Run it on a real model

The same router and spec work unchanged against a real model. This example
uses the Hugging Face adapter with GPT-2 (about 500 MB, downloaded on first
use) on the CPU. The router is built without a registry, so it finds
`norm_threshold` and `norm_ema` in the global registry where sections 1 and 4
put them.

The adapter calls `end_request` itself when generation stops, so collect the
results with an `on_request_end` listener:

```python
from undercurrent.adapters.hf import HFEngineAdapter

adapter = HFEngineAdapter()
adapter.load_model("gpt2", device="cpu")

router = Router()  # the global registry: @probe / @register_probe names, plus plugins
results_by_request = {}
router.on_request_end(lambda request_id, results: results_by_request.__setitem__(request_id, results))

GENERATION = {"max_new_tokens": 20, "do_sample": False, "pad_token_id": 50256}
PROMPT = "The quick brown fox"

adapter.register_extraction("run-1", list(spec))
text = adapter.generate("run-1", PROMPT, GENERATION, router)
results = results_by_request["run-1"]
print(repr(text))
print("gate: ", results["gate"].verdict)
print("drift:", results["drift"].verdict)
```

GPT-2's layer-6 norms for these tokens stay below 100, so neither probe trips
and all 20 tokens are generated. Lower the gate's threshold through
`probe_args` and the same probe stops generation part-way through:

```python
from undercurrent.spec import parse_dict, probe_spec_to_dict

strict_spec_dict = probe_spec_to_dict(spec)
for point in strict_spec_dict["extraction_points"]:
    if point["name"] == "gate":
        point["probe_args"]["threshold"] = 95.0
strict_spec = parse_dict(strict_spec_dict)

adapter.register_extraction("run-2", list(strict_spec))
short_text = adapter.generate("run-2", PROMPT, GENERATION, router)
gate = results_by_request["run-2"]["gate"]
print(repr(short_text))
print("gate:", gate.verdict)
print("stopped by:", gate.signal_history[-1].metadata)
assert gate.verdict["flagged"] is True
assert len(short_text) < len(text)

router.shutdown()
```

The adapter drives one router for its whole lifetime and runs one `generate`
call at a time. For production serving, the vLLM adapter takes the same spec
and probes; see [Deploy with vLLM](vllm-deployment.md).

!!! tip "The front door: `ProbedModel`"
    The adapter and router are the building blocks underneath `ProbedModel`
    (see the [Quickstart](../getting-started/quickstart.md)). A probe you
    have written and registered here plugs into it the same way, through a
    spec that names its `probe_type`.

### Probes with trained weights

A probe that runs a trained classifier needs its weights. There is no probe
save/load API in v0.1. Load the weights yourself, with `safetensors`, which
`transformers` already depends on. Loading a file on every `spawn` would add
disk I/O to every request, so load once and share the read-only result, for
example:

```py
from functools import lru_cache

import torch
from safetensors.torch import load_file


@lru_cache(maxsize=None)
def load_head(path: str) -> torch.nn.Linear:
    weights = load_file(path)
    head = torch.nn.Linear(weights["weight"].shape[1], weights["weight"].shape[0])
    head.load_state_dict(weights)
    return head.eval()


@probe("toxicity", threshold=0.8)
def toxicity(record, *, weights: str = "toxicity_head.safetensors") -> float:
    with torch.inference_mode():
        return load_head(weights)(record.tensor.float()).sigmoid().item()
```

The [`examples/train_probe/`](https://github.com/wrynx/undercurrent/tree/main/examples/train_probe)
example trains a linear probe on GPT-2 activations, saves it as
`safetensors`, and loads it inside a probe. For a bigger worked example (a
single-shot and a trajectory safety probe bound to a spec, plus an MLP probe
for Llama), read the
[content-safety demo](https://github.com/wrynx/undercurrent/tree/main/examples/content_safety).

## 8. Isolation rules and pitfalls

**State belongs to the instance.** A probe class is a stateless factory. The
router spawns a fresh instance per `(request, extraction point)` and discards
it after `on_end`, so anything on `self` is automatically per-request. Put
mutable state on `self`, in `__init__` or `on_start`. Undercurrent enforces
part of this: a class-level `list`, `dict`, `set` or `bytearray` is rejected
when the class is defined, because every instance would share it.

```python
try:

    class LeakyProbe(Probe):
        probe_kind = "trajectory"
        seen = []  # shared by every request: rejected

except TypeError as exc:
    print(exc)
```

**No global mutation.** The same goes for module-level state: a global
counter, a cache keyed by request, a list of "recent" activations. Two
requests running at once would see each other's data. Shared objects should be
read-only after loading, like the trained head above. If a probe must report
something across requests, emit it in the `ProbeResult` or a signal's
`metadata` and aggregate downstream, in a sink or an `on_request_end`
listener (see [Observation sinks](observation-sinks.md)).

**Thread safety in async mode.** At an `execution_mode: async` point, each
probe instance gets its own worker thread that calls `on_activation` in
order, one record at a time, so the instance's own state needs no locks. But
different instances, of the same class or not, run on different threads at
the same time. Anything they share (a model, a tokenizer, a client) must be
safe to use concurrently. Under an inline `block_until_signal` intervention
policy, a call that exceeds `timeout_ms` keeps running in the background while
the router moves on, so the next `on_activation` on the same instance can
overlap it; keep such probes well inside their timeout. See
[Intervention policies](../production/intervention-policies.md) and
[Async execution](../production/async-execution.md).

**Inline probes are on the critical path.** An inline point's `on_activation`
runs on the generation thread, synchronously, for every matching token.
Whatever it costs is added to the latency of that decode step. Keep inline
probes to a few vector operations or one small matrix multiply, and measure
them:

```python
import time

timing_probe = norm_threshold.spawn("bench", "gate")
timing_record = make_record("bench", "gate", 0, 90.0)
start = time.perf_counter()
for _ in range(1000):
    timing_probe.on_activation(timing_record)
per_call_us = (time.perf_counter() - start) / 1000 * 1e6
print(f"norm_threshold.on_activation: {per_call_us:.1f} µs per call")
```

Move anything heavier to an async trajectory point (observe only), narrow the
point's `position` and `layers` so it matches fewer tokens, or use a `stride`.
Also keep in mind:

- An exception raised by an inline probe propagates out of `route()`, and with
  the HF adapter out of `generate()`. At async points, the router logs the
  exception and records a `continue` signal with `router_error` metadata
  instead, so a failing observer never breaks generation.
- `on_end` is called even after an abort, and even when no activation
  arrived. Don't assume `on_activation` ran: initialise in `on_start`, as the
  examples above do.
- Return Python scalars from function probes (`float(...)`, `.item()`), not
  0-d tensors; anything else is a `TypeError`.
- Call `super().__init__()` in your constructor. `spawn` relies on it.

## 9. Ship it as a plugin

When the probe is ready, package it so that `pip install` is all anyone needs.
Undercurrent discovers probes from installed packages through the
`undercurrent.probes` entry-point group: the entry-point **name** is the
`probe_type`, and the **value** is `module:Class`. A function probe works too,
because `@probe` turns the function into a class.

Lay the package out like this:

```text
undercurrent-myprobe/
├── pyproject.toml
├── src/undercurrent_myprobe/
│   ├── __init__.py
│   └── probes.py        # NormEMAProbe and norm_threshold from this guide
└── tests/
    └── test_probes.py   # the router tests from section 6
```

and declare the entry points in `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "undercurrent-myprobe"
version = "0.1.0"
description = "Activation-norm probes for Undercurrent"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
dependencies = ["undercurrent>=0.1,<0.2"]

[project.optional-dependencies]
test = ["pytest"]

[project.entry-points."undercurrent.probes"]
norm_ema = "undercurrent_myprobe.probes:NormEMAProbe"
norm_threshold = "undercurrent_myprobe.probes:norm_threshold"

[tool.hatch.build.targets.wheel]
packages = ["src/undercurrent_myprobe"]
```

After `pip install undercurrent-myprobe`, a spec with `probe_type: norm_ema`
works with a default `Router()` without importing anything, and
`list_probes()` and `undercurrent validate --check-probes` see the new names.
Plugins load lazily: the first time a name is looked up and isn't registered
yet, the matching entry point is imported.

A few rules for plugin modules:

- **Don't register on import.** Let the entry point do it: define classes
  without `@register_probe`, and write function probes as
  `@probe("norm_threshold", threshold=150.0, register=False)`. Registering at
  import time under a different name than the entry point registers the probe
  twice.
- **Explicit registrations win.** If an application registers its own probe
  under a name your plugin also declares, the application's probe is used.
  If two installed packages declare the same name, the first one found is
  used and a warning is logged, so choose distinctive names (a prefix such as
  `myorg_` helps).
- **A broken plugin is skipped, not fatal.** If the module fails to import, or
  the value isn't a concrete `Probe` subclass with a valid `probe_kind`, the
  registry logs a warning and carries on. Test discovery in your own CI:

```py
from undercurrent.core.registry import ProbeRegistry


def test_entry_points_resolve():
    registry = ProbeRegistry()  # fresh: discovers installed plugins only
    assert registry.get("norm_ema").probe_cls.probe_kind == "trajectory"
    assert registry.get("norm_threshold").probe_cls.probe_kind == "single_shot"
```

Publish the package to PyPI under your own name, and tell us about it in a
GitHub discussion so others can find it. If you would like the probe to ship
with Undercurrent itself, see [Contributing](../about/contributing.md).

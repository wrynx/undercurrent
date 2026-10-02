# Probes & lifecycle

<!-- owner: p3-concepts -->

A **probe** is the code that looks at activations. It receives the
`ActivationRecord`s that match its [extraction point](extraction-points.md),
can answer each one with a `ProbeSignal` (carry on, flag, or abort), and
produces one `ProbeResult` (its verdict) when the request ends.

## The simplest probe: a function

For a stateless check that only needs to look at one activation at a time, a
plain function decorated with `@probe(...)` is enough. It takes the record and
returns a score (`float`), a decision (`bool`), a full `ProbeSignal`, or `None`.
Undercurrent wraps it in a probe class for you.
[Write a custom probe](../guides/custom-probe.md) covers the decorator and how
return values map to signals.

Use a class, as described below, when the probe needs per-request state (a
running score, a history of activations) or custom setup and teardown.

## The lifecycle

A probe class subclasses `undercurrent.core.Probe` and implements three
methods. For each request, the router calls them in this order:

```py
probe = MyProbe.spawn(request_id, extraction_point_name)  # a fresh instance
probe.on_start(request_ctx)  # once
for record in matching_activations:  # zero or more times
    signal = probe.on_activation(record)  # -> ProbeSignal | None
result = probe.on_end(request_ctx)  # once, always -> ProbeResult
```

| Method | Called | Returns |
| --- | --- | --- |
| `on_start(request_ctx)` | Once, before any activation | nothing |
| `on_activation(record)` | Once per matching activation, in order | a `ProbeSignal`, or `None` if there's nothing to say |
| `on_end(request_ctx)` | Exactly once, when generation ends or is aborted | a `ProbeResult` |

`on_end` must return a `ProbeResult` even if `on_activation` was never called,
for example because the extraction point never matched a token.

Here is a complete class probe, driven by hand with a synthetic activation:

```python
from undercurrent.core import (
    ActivationRecord,
    Probe,
    ProbeAction,
    ProbeResult,
    ProbeSignal,
    RequestContext,
)


class NormProbe(Probe):
    """Flags an activation whose L2 norm exceeds a threshold."""

    probe_kind = "single_shot"

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__()  # required: sets up request_id / extraction_point_name
        self.threshold = threshold
        self.norm = None
        self.signals = []

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def on_activation(self, record: ActivationRecord):
        self.norm = sum(x * x for x in record.tensor) ** 0.5
        if self.norm <= self.threshold:
            return None
        signal = ProbeSignal(action=ProbeAction.FLAG, confidence=self.norm, metadata={"norm": self.norm})
        self.signals.append(signal)
        return signal

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=self.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"norm": self.norm, "flagged": bool(self.signals)},
            signal_history=self.signals,
        )


ctx = RequestContext(request_id="req-1", prompt_metadata={}, extraction_point_config=None)
probe = NormProbe.spawn("req-1", "norm_check", threshold=1.0)
probe.on_start(ctx)
signal = probe.on_activation(
    ActivationRecord(
        request_id="req-1",
        extraction_point_name="norm_check",
        layer=3,
        token_pos=7,
        tensor_type="residual_stream",
        tensor=[0.6, 0.8, 0.9],
        is_generated=False,
    )
)
assert signal.action is ProbeAction.FLAG
result = probe.on_end(ctx)
assert result.verdict["flagged"]
```

In normal use you never call these methods yourself. You register the class
with a router (or with `ProbedModel`), and the router spawns and drives one
instance per request.

## Isolation model

The **class** is a reusable factory, registered once and shared by every
request. An **instance** is cheap and single-use: one is spawned for each
`(request_id, extraction_point_name)` pair, used for that one request, and
thrown away. Instances are never shared across requests, extraction points or
threads, so per-request state on `self` is safe without locks.

`Probe.spawn(request_id, extraction_point_name, **probe_kwargs)` is the way to
create an instance. It passes `probe_kwargs` to your `__init__` and stamps the
instance with its request and extraction point. An instance created by calling
the class directly raises `RuntimeError` when you read `.request_id`.

Undercurrent enforces the isolation rule: defining a mutable class attribute
(`list`, `dict`, `set`, `bytearray`) on a probe subclass raises `TypeError` at
class-definition time, because every instance would silently share it.

```python
try:

    class LeakyProbe(Probe):
        probe_kind = "trajectory"
        seen = []  # shared by every instance -- rejected

except TypeError as exc:
    print(exc)
```

Put mutable state on `self` in `__init__` or `on_start` instead.

`ProbeFactory(probe_cls, probe_kwargs)` pairs a probe class with fixed
constructor arguments. It is what a router's `probe_registry` maps each
`probe_type` to:

```python
from undercurrent.core import ProbeFactory

factory = ProbeFactory(NormProbe, {"threshold": 2.5})
instance = factory.spawn("req-2", "norm_check")
assert instance.threshold == 2.5 and instance.request_id == "req-2"
```

An extraction point's `probe_args` override the factory's arguments for the
same keys, so one registered probe can run with different settings at
different points.

## Single-shot and trajectory probes

Every probe class declares a `probe_kind`, and every extraction point repeats
it as `probe_kind` in the spec:

| `probe_kind` | Looks at | State | Typical position | Execution modes |
| --- | --- | --- | --- | --- |
| `single_shot` | One activation | None to speak of: decides from that one activation | A single token, e.g. `prompt[-1]` | `inline` only |
| `trajectory` | A stream of activations over the generation | Accumulated across calls (running score, history) | Continuous, e.g. `generated[*]` | `inline` or `async` |

A **single-shot** probe puts its decision logic in `on_activation`; `on_end`
just packages what it computed (with a sensible empty verdict if it never ran).
The content-safety demo's `SingleTokenSafetyProbe` is this shape: score one
activation with a classifier head and flag it above a threshold.

A **trajectory** probe updates its state on every activation and can intervene
part-way through, for example by returning `abort` the first time a running
score crosses a threshold. Its verdict is only final in `on_end`. Because the
router feeds an instance its activations one at a time and in order (for async
points too), the probe can rely on that order.

Only trajectory probes may run `async`: an async probe works through a queue
of activations in the background, which only makes sense for a probe that
accumulates state. The spec parser and the router both reject `async` with
`single_shot`.

`undercurrent.core.examples` has two small reference probes:
`MLPClassifierProbe` (single-shot) and `TrajectoryScoreProbe` (trajectory, a
running mean with abort-on-threshold). Both use deterministic placeholder
scoring rather than trained models:

```python
from undercurrent.core.examples import TrajectoryScoreProbe

traj = TrajectoryScoreProbe.spawn("req-3", "running_mean", threshold=0.5)
traj.on_start(ctx)
signals = []
for pos, values in enumerate([[0.1, 0.2], [0.4, 0.6], [0.9, 1.0]]):
    record = ActivationRecord("req-3", "running_mean", 4, 10 + pos, "mlp_out", values, True)
    signals.append(traj.on_activation(record))

assert signals[0] is None  # running mean still below 0.5
assert signals[2].action is ProbeAction.ABORT  # crossed the threshold
print(traj.on_end(ctx).verdict)  # {'final_mean': ..., 'count': 3, 'aborted': True}
```

The full content-safety demo, with single-shot and trajectory probes backed by
small PyTorch heads, lives in `examples/content_safety/` in the repository.

## Core types

**`ActivationRecord`** (`undercurrent.spec`, re-exported from
`undercurrent.core`): one captured activation.

| Field | Meaning |
| --- | --- |
| `request_id` | The request it belongs to |
| `extraction_point_name` | The extraction point that matched |
| `layer` | The single layer it came from (one record per layer) |
| `token_pos` | Absolute token index in prompt + generated sequence |
| `tensor_type` | e.g. `"residual_stream"` (a plain string) |
| `tensor` | The activation itself. The shipped adapters pass a 1-D CPU `torch.Tensor` of size `hidden_dim`; synthetic records can use any array-like. |
| `is_generated` | `False` for prompt tokens |
| `timestamp` | Capture time (seconds since the epoch) |

`record.metadata()` returns every field except `tensor` as a dict, which is
handy for logging.

**`ProbeSignal`** (`undercurrent.core`): what `on_activation` may return.

| Field | Meaning |
| --- | --- |
| `action` | `ProbeAction.CONTINUE` (default), `FLAG` or `ABORT` |
| `metadata` | Probe-defined dict: scores, intermediate values |
| `confidence` | Optional float, meaning probe-defined |
| `timestamp` | Creation time |

`ABORT` from an inline extraction point stops generation (see
[Interventions](interventions.md)). `FLAG` and `CONTINUE` never change
generation; they are there to be logged and inspected.

**`ProbeResult`** (`undercurrent.core`): what `on_end` returns, exactly once.

| Field | Meaning |
| --- | --- |
| `request_id`, `extraction_point_name` | Which request and point it belongs to |
| `verdict` | The probe's conclusion. Any shape; document yours. |
| `signal_history` | The signals the probe chose to record, in order (the probe fills this in) |
| `metadata` | Extra probe-defined data (timing, debug info) |

**`RequestContext`** (`undercurrent.core`): passed to `on_start` and `on_end`.
It is frozen, so it can't be used to stash state.

| Field | Meaning |
| --- | --- |
| `request_id` | The request |
| `prompt_metadata` | Adapter-defined dict. The shipped adapters include `model`, `prompt` and `prompt_len`. |
| `extraction_point_config` | Optional; the resolved extraction point when the caller supplies it (the shipped adapters pass `None`) |

## Failures

A probe that raises doesn't take generation down with it:

- **Async** points: the exception is logged, counted in the
  [metrics](../production/metrics.md) as a probe error, and turned into a
  `ProbeSignal(action=CONTINUE)` whose metadata has `router_error: True` and
  the error message, which is forwarded to the log sink. The worker moves on
  to the next activation.
- **Inline** points under `intervention.mode: block_until_signal`: the
  exception is treated like a timeout and replaced by the `on_timeout`
  fallback signal (see [Intervention policies](../production/intervention-policies.md)).
- **Inline** points under the default `reject` mode: the exception propagates
  out of `router.route()` to the engine adapter. Catch your own errors in
  `on_activation` if you want to fail open.

## Next

- [Execution modes: inline vs async](execution-modes.md)
- [Write a custom probe](../guides/custom-probe.md)
- [`undercurrent.core` API reference](../reference/core.md)

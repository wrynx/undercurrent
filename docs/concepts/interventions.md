# Interventions

<!-- owner: p3-concepts -->

An **intervention** is an inline probe changing the course of generation. In
v0.1 there is one intervention: **abort**. When an inline probe returns
`ProbeSignal(action=ProbeAction.ABORT)`, the engine adapter stops generating
for that request. The router still finalizes every probe, so you get each
probe's `ProbeResult` for the truncated generation.

The **intervention policy** of an extraction point decides how long generation
may wait for the probe's answer.

## Intervention policy

```yaml
intervention:
  mode: block_until_signal
  timeout_ms: 250
  on_timeout: abort
```

| Field | Values | Meaning |
| --- | --- | --- |
| `mode` | `reject` (default) \| `block_until_signal` | `reject`: call the probe and use whatever it returns, with no timeout. `block_until_signal`: wait at most `timeout_ms` for the probe's answer. |
| `timeout_ms` | integer `>= 1` | Required with `block_until_signal`, forbidden with `reject`. |
| `on_timeout` | `continue` (default) \| `abort` | What to do if the probe doesn't answer in time or raises. Only allowed with `block_until_signal`. |

Despite the name, `reject` doesn't reject anything. It means "no wait
contract": the probe runs synchronously and generation waits for however long
it takes. `block_until_signal` puts a hard bound on that wait. Only inline
extraction points can use `block_until_signal`; async points can't intervene
at all.

You can set a policy per extraction point in the spec, or a router-wide
default for every point that doesn't set its own:

```py
from undercurrent.router import Router
from undercurrent.spec import InterventionMode, InterventionPolicy, TimeoutAction

router = Router(
    probe_registry,
    default_intervention_policy=InterventionPolicy(
        mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=250, on_timeout=TimeoutAction.ABORT
    ),
)
```

A point's own `intervention` always wins over the router default.

## What happens on a timeout

If a `block_until_signal` probe doesn't answer within `timeout_ms`, or raises,
the router doesn't wait any longer. It returns a stand-in signal with action
`abort` (if `on_timeout: abort`) or `continue` (if `on_timeout: continue`), and
`metadata["intervention_fallback"] = True`:

```python
import time

from undercurrent.core import Probe, ProbeAction, ProbeFactory, ProbeResult, ProbeSignal, RequestContext
from undercurrent.router import Router
from undercurrent.spec import (
    ActivationRecord,
    ExecutionMode,
    ExtractionPoint,
    InterventionMode,
    InterventionPolicy,
    ProbeKind,
    TensorType,
    TimeoutAction,
    parse_position,
)


class SlowProbe(Probe):
    probe_kind = "single_shot"

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        time.sleep(0.2)  # far slower than the 20 ms budget below
        return ProbeSignal(action=ProbeAction.CONTINUE)

    def on_end(self, request_ctx):
        return ProbeResult(self.request_id, self.extraction_point_name, verdict=None)


point = ExtractionPoint(
    name="guard",
    layers=(2,),
    tensor_type=TensorType.RESIDUAL_STREAM,
    position=parse_position("prompt[-1]"),
    stride=None,
    until=None,
    probe_type="slow",
    probe_kind=ProbeKind.SINGLE_SHOT,
    execution_mode=ExecutionMode.INLINE,
    queue_depth=None,
    intervention=InterventionPolicy(
        mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=20, on_timeout=TimeoutAction.ABORT
    ),
)

router = Router(probe_registry={"slow": ProbeFactory(SlowProbe)})
router.register_request("req-1", [point], RequestContext("req-1", {}, None))

start = time.monotonic()
signal = router.route(ActivationRecord("req-1", "guard", 2, 5, "residual_stream", [0.0], False))
waited_ms = (time.monotonic() - start) * 1000

assert signal.action is ProbeAction.ABORT  # on_timeout: abort
assert signal.metadata["intervention_fallback"] and signal.metadata["reason"] == "timeout"
assert waited_ms < 200  # route() returned after ~20 ms, not 200
router.end_request("req-1")
router.shutdown()
```

Pick `on_timeout` by asking what a missing answer should mean:

- `continue` **fails open**: a slow or broken probe never blocks output.
  Right for monitoring and quality probes.
- `abort` **fails closed**: no answer, no output. Right for safety gates where
  letting unchecked text through is worse than truncating it.

The slow call isn't cancelled. Python can't interrupt a running thread, so it
finishes in the background and its result is discarded.

## Circuit breaker

A probe that keeps timing out costs up to `timeout_ms` on every matching
token. To cap that, the router counts **consecutive** timeouts and exceptions
per extraction point name, across requests. When the count reaches
`circuit_breaker_threshold` (default 5), the router **trips** the breaker for
that extraction point: from then on it runs as plain `reject`, with no timeout
and no fallback, for the rest of the router's life. Any success before then
resets the count.

A trip is logged as a warning and, if a log sink is attached, reported as a
`ProbeSignal(action=FLAG)` with `metadata["circuit_breaker_tripped"] = True`.
Details and tuning advice are in
[Intervention policies & timeouts](../production/intervention-policies.md#circuit-breaker).

## What each engine adapter supports

Every adapter honours an inline `abort`, and the router enforces `timeout_ms`
the same way for all of them. What differs is **what the wait blocks** and
**how quickly an abort lands**.

| | Hugging Face (`adapters.hf`) | vLLM (`adapters.vllm`) |
| --- | --- | --- |
| Inline `abort` | Stops before the next token | Stops at the next scheduler step; expect one or a few extra tokens |
| `reject` mode | Supported | Supported |
| `block_until_signal` | **Supported and safe**: the wait only delays this one request | **Avoid under concurrency**: the wait blocks the shared engine thread and stalls every request in the batch. Logs a one-time warning. |
| Recommended for blocking gates | `block_until_signal` with a `timeout_ms` | `reject` with fast probes, or `async` to observe only |

**Hugging Face.** The adapter runs one sequential decode loop for one request
at a time. The forward hook calls `route()` on the decode thread, and a
`StoppingCriteria` checks the abort flag before the next token. A bounded wait
therefore delays exactly the request that asked for it, and nothing else.

**vLLM.** Continuous batching runs many requests' tokens in one forward pass.
The hook runs on vLLM's shared engine thread, so a `block_until_signal` wait,
though still capped at `timeout_ms`, stalls the whole step for every request in
it. Aborts go through the engine's own `abort()`, which takes effect when the
next step is scheduled. The adapter warns once if an extraction point uses
`block_until_signal`; it can't detect a router-wide default, so don't set one
for vLLM deployments. See
[The vLLM adapter](../internals/vllm-adapter.md) and
[Deploy with vLLM](../guides/vllm-deployment.md).

## Next

- [Intervention policies & timeouts](../production/intervention-policies.md):
  latency cost, circuit-breaker tuning and recommended production settings.
- [Execution modes](execution-modes.md): when to observe instead of intervene.
- [`undercurrent.spec`](../reference/spec.md) and
  [`undercurrent.router`](../reference/router.md) API reference.

# Intervention policies & timeouts

<!-- owner: p3-concepts -->

This page goes deep on how inline probes are allowed to hold up generation:
the two intervention modes, timeouts and fallbacks, the circuit breaker, what
each engine adapter does with them, and what to run in production. For the
short version, see [Interventions](../concepts/interventions.md).

## The two modes

An extraction point's effective policy is its own `intervention` if the spec
sets one, otherwise the router's `default_intervention_policy`, otherwise
`InterventionPolicy()` (mode `reject`). The router resolves it once, when the
request registers.

### `reject` (default)

```yaml
intervention:
  mode: reject     # or just leave `intervention` out
```

`route()` calls `on_activation` directly on the caller's thread and returns
whatever it produces. There is no timeout and no fallback. If the probe takes
80 ms, that token waits 80 ms; if it hangs, generation hangs; if it raises, the
exception propagates out of `route()` into the engine adapter.

This is the lowest-overhead mode (no thread hand-off), and it is the right
choice when the probe is fast and well-behaved: a linear probe or small MLP
head on an activation that is already on the CPU.

### `block_until_signal`

```yaml
intervention:
  mode: block_until_signal
  timeout_ms: 50
  on_timeout: continue
```

`route()` submits `on_activation` to the router's shared thread pool and waits
for the result for at most `timeout_ms`:

- **Answer in time**: the probe's signal is returned unchanged.
- **Timeout**: `route()` returns a fallback signal immediately.
- **Exception**: logged with its traceback; `route()` returns a fallback
  signal.

The fallback signal's `action` is `abort` if `on_timeout: abort`, else
`continue`. Its `metadata` carries:

| Key | Value |
| --- | --- |
| `intervention_fallback` | `True` |
| `reason` | `"timeout"` or `"exception"` |
| `extraction_point_name` | the point's name |
| `timeout_ms` | the configured budget |
| `error_type`, `error` | only when `reason` is `"exception"` |

A late answer is **not** cancelled: the timed-out call keeps running in the
background and its result is thrown away. For a trajectory probe this has a
consequence: if generation continues, the next activation's call can start
while the previous one is still running, on another thread, on the **same**
probe instance. Either keep `on_activation` well under `timeout_ms`, or make
the probe's state safe against that overlap.

`block_until_signal` is inline-only. The parser rejects it on async points,
and the router rejects it at registration, including when it would be
inherited from a router-wide default.

### Setting a router-wide default

```py
from undercurrent.router import Router
from undercurrent.spec import InterventionMode, InterventionPolicy, TimeoutAction

router = Router(
    probe_registry,
    default_intervention_policy=InterventionPolicy(
        mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=50, on_timeout=TimeoutAction.CONTINUE
    ),
    circuit_breaker_threshold=5,
)
```

A router-wide `block_until_signal` default also applies to async points that
don't set their own policy, and the router will refuse to register them.
Give those points an explicit `intervention: {mode: reject}` in the spec, or
keep the router default at `reject` and opt in per point. The second is
usually clearer.

## Choosing `timeout_ms` and `on_timeout`

`timeout_ms` is a per-activation budget, not a per-token or per-request one.
Each matching `(extraction point, layer)` pair is a separate `route()` call, and
an adapter makes them one after another. The worst-case added latency for one
token is therefore:

```text
sum over blocking points matching this token of  (layers in the point × timeout_ms)
```

A point with `layers: [8, 16, 24]` and `timeout_ms: 50` can add up to 150 ms to
a token in the worst case, and with `position: "generated[*]"` that can happen
on every token until the circuit breaker steps in.

Guidelines:

- Measure the probe's latency distribution on representative inputs and set
  `timeout_ms` comfortably above its p99. The timeout exists to catch a sick
  probe, not to clip a healthy one.
- Remember the thread hand-off: under load, the call may first wait for a free
  pool thread (see [Async execution](async-execution.md#bindings-pin-threads)).
  That wait counts against `timeout_ms`.
- `on_timeout: continue` fails open. Use it for monitoring, quality and
  steering-detection probes, where a missing answer should not cost the user
  their response.
- `on_timeout: abort` fails closed. Use it for gates where unchecked output is
  worse than truncated output. Pair it with a probe you've load-tested, since
  every timeout becomes a user-visible truncation.

## Circuit breaker

A probe that is systematically too slow or broken costs up to `timeout_ms` on
every matching token, on every request. The circuit breaker stops that.

- The router counts **consecutive** failures (timeouts and exceptions) under
  `block_until_signal`, per **extraction point name**, across all requests.
  This is the one piece of router state that is shared across requests,
  deliberately: one unlucky call doesn't trip it, a pattern does.
- A success resets the count to zero.
- When the count reaches `circuit_breaker_threshold` (default 5), the breaker
  **trips**. From then on, every `route()` for that extraction point name, for
  requests already running and requests registered later, is dispatched as
  plain `reject`: the probe still runs and its real signal is still returned,
  but with no timeout, no fallback and no thread hand-off.
- A trip is permanent for the life of the `Router`. There is no automatic
  half-open retry and no reset method; create a new router (for example by
  restarting the server) to re-enable the blocking policy.
- A trip is logged as a warning on the `undercurrent.router` logger and, if a
  log sink is attached, written as a signal with `action=flag` and metadata
  `{"circuit_breaker_tripped": True, "extraction_point_name": ..., "consecutive_failures": ..., "reason": ...}`.

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


class SlowFlagProbe(Probe):
    probe_kind = "single_shot"

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        time.sleep(0.1)
        return ProbeSignal(action=ProbeAction.FLAG, metadata={"real": True})

    def on_end(self, request_ctx):
        return ProbeResult(self.request_id, self.extraction_point_name, verdict=None)


class ListSink:
    def __init__(self):
        self.signals = []

    def write_signal(self, request_id, extraction_point_name, signal):
        self.signals.append(signal)

    def write_result(self, request_id, extraction_point_name, result):
        pass


point = ExtractionPoint(
    name="gate",
    layers=(1,),
    tensor_type=TensorType.RESIDUAL_STREAM,
    position=parse_position("prompt[-1]"),
    stride=None,
    until=None,
    probe_type="slow_flag",
    probe_kind=ProbeKind.SINGLE_SHOT,
    execution_mode=ExecutionMode.INLINE,
    queue_depth=None,
    intervention=InterventionPolicy(
        mode=InterventionMode.BLOCK_UNTIL_SIGNAL, timeout_ms=10, on_timeout=TimeoutAction.CONTINUE
    ),
)
router = Router(probe_registry={"slow_flag": ProbeFactory(SlowFlagProbe)}, circuit_breaker_threshold=2)
sink = ListSink()
router.attach_log_sink(sink)


def one_request(request_id):
    router.register_request(request_id, [point], RequestContext(request_id, {}, None))
    signal = router.route(ActivationRecord(request_id, "gate", 1, 4, "residual_stream", [0.0], False))
    router.end_request(request_id)
    return signal


first, second = one_request("req-1"), one_request("req-2")
assert first.metadata["intervention_fallback"] and second.metadata["intervention_fallback"]
assert sink.signals[0].metadata["circuit_breaker_tripped"]  # tripped after 2 consecutive timeouts

third = one_request("req-3")  # now plain `reject`: waits the full 100 ms, real answer
assert third.metadata == {"real": True}
router.shutdown()
```

Choosing the threshold: lower values protect latency faster but risk tripping
on a short blip, such as a GC pause or a cold cache on the first requests.
The default of 5 suits most cases. Monitor for trips: after one, the probe runs
with no time bound at all, so a probe that went from slow to hung would hang
generation. If that risk matters more than the probe's verdict, alert on the
trip and restart, or move the probe to `async`.

## Engine adapters

The router enforces `timeout_ms` identically for every adapter. Adapters
differ in what a wait blocks and how soon an abort takes effect.

### Hugging Face (`undercurrent.adapters.hf`)

The adapter drives one sequential `model.generate()` loop for one request at a
time. Forward hooks call `route()` synchronously on the decode thread; if an
inline point returns `abort`, the hook sets a flag that a `StoppingCriteria`
checks before the next token.

- **Abort latency:** generation stops before the next token. All hooks for the
  current forward pass still run first, so other points see the current token.
- **`block_until_signal`:** fully supported. The wait delays only this
  request, because the decode step is the whole engine for this request.
- **Latency cost:** each matching activation's probe time (or up to
  `timeout_ms`) is added directly to that token's latency.
- **Limits:** one `generate()` call at a time per adapter, no beam search or
  batch size above 1.

### vLLM (`undercurrent.adapters.vllm`)

vLLM runs continuous batching: one forward pass carries tokens from many
requests. The adapter's forward hooks run on vLLM's shared engine thread and
call `route()` there.

- **Abort latency:** an inline `abort` adds the request to a pending set; the
  adapter cancels it through the engine's own `abort()` after the current
  step. vLLM has already scheduled that step, so expect at least one extra
  token, and a few under heavy load.
- **`block_until_signal`:** still bounded by `timeout_ms`, but the wait blocks
  the shared engine thread, stalling **every request in the batch**, not only
  the one whose activation is being probed. The adapter logs a one-time
  warning when an extraction point is configured with it. It can't see a
  router-wide default, so that case isn't warned about.
- **Latency cost:** an inline probe's run time is paid once per matching
  activation, on the engine thread, by the whole batch. In `reject` mode keep
  inline probes very cheap.
- **Async points** are where vLLM deployments should put anything that isn't
  cheap: the hook only enqueues.

See [The vLLM adapter](../internals/vllm-adapter.md) for the internals and
[Deploy with vLLM](../guides/vllm-deployment.md) for deployment.

## Recommended production settings

| Situation | Execution mode | Intervention | Notes |
| --- | --- | --- | --- |
| Monitoring, analytics, data collection | `async` | `reject` (implied) | Nothing on the generation path. Size queues and pool per [Async execution](async-execution.md). |
| Safety gate, HF backend | `inline` | `block_until_signal`, `timeout_ms` above probe p99, `on_timeout: abort` | Fails closed; the wait only affects the gated request. Alert on circuit-breaker trips. |
| Quality or steering probe that may intervene, HF backend | `inline` | `block_until_signal`, `on_timeout: continue` | Fails open; bounded latency. |
| Gate on vLLM with concurrent traffic | `inline` | `reject` | Keep the probe as cheap as possible (its run time is paid by the whole batch); expect an abort to land a step or more late. |
| Anything expensive on vLLM | `async` | `reject` (implied) | Observe only; act on results after the request (for example, filter the response in your server). |
| Single-request or offline vLLM runs | `inline` | `block_until_signal` acceptable | No batch to stall. |

Also:

- Keep the router-wide `default_intervention_policy` at `reject` and opt in per
  extraction point, so async points and vLLM deployments don't inherit a
  blocking policy by accident.
- Probe on as few layers as you need: each layer of a blocking point is
  another `timeout_ms` of worst-case latency per token.
- Make `on_activation` exception-safe in `reject` mode; an exception there
  ends the generation.
- Watch the `undercurrent.router` logger (or your log sink) for
  `circuit_breaker_tripped` and `intervention_fallback`.

## Related

- [Interventions](../concepts/interventions.md)
- [Execution modes: inline vs async](../concepts/execution-modes.md)
- [Async execution & backpressure](async-execution.md)
- [Observation sinks](../guides/observation-sinks.md)
- [Metrics](metrics.md)
- [Embed in your serving stack](embedding.md)
- [`undercurrent.router`](../reference/router.md) and
  [`undercurrent.spec`](../reference/spec.md) API reference

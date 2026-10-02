# probing_core

The **probe interface layer** for the activation-probing platform.

This package defines how a probe plugs into the platform — the `Probe`
lifecycle, the isolation model between spawned instances, and the
`ProbeSignal` / `ProbeResult` data types — and nothing about how
activations get to a probe or which probe runs where. Those are the
router's and adapters' jobs, built on top of this package.

It depends on [`undercurrent.spec`](probing_spec.md) for `ActivationRecord`
(imported directly, not redefined — see [Dependency on
probing_spec](#dependency-on-probing_spec)). It implements no router and no
inference-engine adapter.

## Install

```bash
pip install -e .   # from the repo root; installs undercurrent (spec + core)
```

Requires Python 3.9+.

## The Probe lifecycle

```python
from undercurrent.core import Probe, ProbeAction, RequestContext

probe = SomeProbeSubclass.spawn(request_id, extraction_point_name)  # fresh instance
probe.on_start(RequestContext(request_id, prompt_metadata, extraction_point_config))
for record in matching_activations:            # zero or more calls
    signal = probe.on_activation(record)
    if signal is not None and signal.action is ProbeAction.ABORT:
        break
result = probe.on_end(request_ctx)              # always called, always returns a ProbeResult
```

`on_end` is guaranteed to return a `ProbeResult` even if `on_activation` was
never called — e.g. its extraction point never matched a token during this
request.

## Isolation model

A `Probe` **subclass** is a stateless, reusable factory: register it once
(e.g. in a probe registry keyed by `probe_type`) and reuse it across every
request. A `Probe` **instance** is cheap and single-use, scoped to exactly
one `(request_id, extraction_point_name)` pair, created via:

```python
probe = MyProbe.spawn(request_id, extraction_point_name, **probe_kwargs)
```

`spawn()` is the only sanctioned way to construct one for real use.
`probe_kwargs` are forwarded to `__init__` (e.g. a threshold) and are fresh
per call. Instances must never be reused or shared across requests,
extraction points, or concurrent calls.

This isn't just convention — `Probe.__init_subclass__` rejects any subclass
that defines a class-level `list`/`dict`/`set`/`bytearray` attribute, since
that's exactly the mechanism by which "isolated" instances end up silently
sharing state:

```python
class BadProbe(Probe):
    probe_kind = "trajectory"
    _cache = {}   # TypeError at class-definition time
```

Put per-request mutable state on `self`, assigned in `__init__` or
`on_start`, instead.

`ProbeFactory` is an optional convenience for callers (e.g. a router) that
want a single value binding a `Probe` subclass to fixed spawn kwargs,
instead of tracking a class and a kwargs dict separately — it's a thin
wrapper around `.spawn()`, not a second mechanism.

## Core types

- **`ProbeSignal`** — `action` (`continue | abort | flag`, default
  `continue`), `metadata: dict`, `confidence: Optional[float]`,
  `timestamp: float`. Returned by `on_activation`, immediately or not at
  all — this is what lets an inline probe intervene mid-generation.
- **`ProbeResult`** — `request_id`, `extraction_point_name`, `verdict: Any`
  (probe-defined — a classifier probe's verdict looks nothing like a
  trajectory probe's), `signal_history: List[ProbeSignal]`,
  `metadata: dict`. Returned by `on_end`, exactly once.
- **`RequestContext`** — `request_id`, `prompt_metadata: dict`,
  `extraction_point_config: Any`. Frozen; passed to `on_start`/`on_end`.

## Example probes

`undercurrent.core.examples` has two reference implementations (deterministic
stub scoring, not real models — they exist to validate the interface):

- **`MLPClassifierProbe`** (`single_shot`) — classifies one activation via
  a stub forward pass; verdict is `{"predicted_class": int, "logits":
  [float, ...]}`.
- **`TrajectoryScoreProbe`** (`trajectory`) — accumulates a running mean
  score across every matching activation; after each one,
  `check_intervention()` emits `action=abort` the first time the running
  mean crosses `threshold`. Verdict (only finalized in `on_end`) is
  `{"final_mean": float, "count": int, "aborted": bool}`.

```python
from undercurrent.core.examples import TrajectoryScoreProbe

probe = TrajectoryScoreProbe.spawn("req-1", "ep-1", threshold=0.8)
probe.on_activation(record_1)   # may return None or a ProbeSignal
probe.on_activation(record_2)
result = probe.on_end(request_ctx)
print(result.verdict)  # {"final_mean": ..., "count": 2, "aborted": ...}
```

## Dependency on probing_spec

The stub-fallback dependency shim was removed in the single-package consolidation; shared types are imported directly.

## Package layout

```
src/undercurrent/core/
  activation.py   ActivationRecord (re-exported from undercurrent.spec)
  signal.py       ProbeAction, ProbeSignal
  result.py       ProbeResult
  context.py      RequestContext
  probe.py        Probe (ABC + spawn isolation model), ProbeFactory
  examples/       MLPClassifierProbe, TrajectoryScoreProbe
tests/core/       unit tests (pytest, at the repo root)
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/core
```

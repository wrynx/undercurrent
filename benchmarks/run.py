#!/usr/bin/env python
"""Undercurrent overhead benchmark harness.

Measures what probing costs compared with plain generation: 0 / 1 / N probes,
inline vs async, on the Hugging Face transformers and vLLM backends, plus an
opt-in backpressure scenario (`async-slow`) that runs a deliberately slow
async probe once per router `OverflowPolicy`.

    python benchmarks/run.py --backend hf --dry-run --out /tmp/bench
    python benchmarks/run.py --backend vllm --model meta-llama/Llama-3.1-8B-Instruct

Every run writes `<out>/<timestamp>-<backend>-<model-slug>/` containing
`env.json`, `results.jsonl` and `summary.csv`. benchmarks/README.md documents
the methodology and the output schema; that schema is a contract (the docs
benchmark page reads these files), so change it only together with
`SCHEMA_VERSION` and the README.

Needs only the standard library plus what `pip install undercurrent` already
brings (torch, transformers); `--backend vllm` also needs vLLM and a CUDA GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

DEFAULT_SCENARIOS = "baseline,inline-1,async-1,inline-8,async-8,mixed"
#: What `--dry-run` runs when `--scenarios` isn't given: every scenario kind.
DRY_RUN_SCENARIOS = "baseline,probed-0,inline-1,async-1,inline-8,async-8,inline-8-mlp,async-8-mlp,mixed,async-slow"
DRY_RUN_MODEL = "hf-internal-testing/tiny-random-gpt2"

#: Defaults for a real (GPU) run; `--dry-run` replaces them with `_DRY_RUN_SIZES`.
_DEFAULT_SIZES = {
    "num_prompts": 64,
    "max_new_tokens": 128,
    "input_len": 128,
    "warmup": 4,
    "repeats": 3,
    "probe_delay_ms": 50.0,
    "slow_queue_depth": 4,
}
_DRY_RUN_SIZES = {
    "num_prompts": 4,
    "batch_size": 1,
    "max_new_tokens": 8,
    "input_len": 16,
    "warmup": 2,
    "repeats": 2,
    "probe_delay_ms": 50.0,
    "slow_queue_depth": 2,
}
DEFAULT_VLLM_BATCH_SIZE = 16
DEFAULT_FLAT_TOLERANCE_PCT = 10.0
MLP_HIDDEN = 256
QUEUE_DEPTH_SERIES_POINTS = 200

log = logging.getLogger("undercurrent.benchmarks")


class HarnessError(Exception):
    """A usage or environment problem, reported as one clear line (no traceback)."""


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeSlot:
    """One extraction point of a scenario."""

    impl: str  # "norm" | "mlp" | "sleep"
    mode: str  # "inline" | "async"


@dataclass(frozen=True)
class Scenario:
    """One unit of work: a probe configuration measured `--repeats` times."""

    name: str
    probed: bool  # False only for `baseline` (no probing machinery at all)
    slots: tuple[ProbeSlot, ...] = ()
    overflow_policy: str | None = None  # async-slow only
    queue_depth: int | None = None  # None: the router's default_queue_depth

    @property
    def num_inline(self) -> int:
        return sum(s.mode == "inline" for s in self.slots)

    @property
    def num_async(self) -> int:
        return sum(s.mode == "async" for s in self.slots)

    @property
    def probe_impls(self) -> str:
        return "+".join(sorted({s.impl for s in self.slots}))

    @property
    def is_slow(self) -> bool:
        return self.overflow_policy is not None


_N_PROBES_RE = re.compile(r"^(inline|async)-(\d+)(-mlp)?$")


def overflow_policy_names() -> list[str]:
    """The router's real `OverflowPolicy` values, in definition order."""
    from undercurrent.router.overflow import OverflowPolicy

    return [p.value for p in OverflowPolicy]


def parse_scenarios(text: str, *, slow_queue_depth: int) -> list[Scenario]:
    """Expand a comma-separated `--scenarios` value into run units.

    Grammar: `baseline`, `probed-0`, `inline-N`, `async-N`, `inline-N-mlp`,
    `async-N-mlp`, `mixed`, `async-slow` (one unit per overflow policy) and
    `async-slow:<policy>` (one policy only).
    """
    policies = overflow_policy_names()
    units: list[Scenario] = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name == "baseline":
            units.append(Scenario("baseline", probed=False))
        elif name == "probed-0":
            units.append(Scenario("probed-0", probed=True))
        elif name == "mixed":
            # 4 inline + 4 async, each group alternating the norm and MLP probes:
            # an inline guard plus background observers, the common production shape.
            slots = tuple(ProbeSlot(("norm", "mlp")[i % 2], "inline") for i in range(4)) + tuple(
                ProbeSlot(("norm", "mlp")[i % 2], "async") for i in range(4)
            )
            units.append(Scenario("mixed", probed=True, slots=slots))
        elif name == "async-slow" or name.startswith("async-slow:"):
            wanted = policies if name == "async-slow" else [name.split(":", 1)[1]]
            for policy in wanted:
                if policy not in policies:
                    raise HarnessError(
                        f"unknown overflow policy {policy!r} in scenario {name!r}; the router's policies are: "
                        f"{', '.join(policies)}"
                    )
                units.append(
                    Scenario(
                        f"async-slow:{policy}",
                        probed=True,
                        slots=(ProbeSlot("sleep", "async"),),
                        overflow_policy=policy,
                        queue_depth=slow_queue_depth,
                    )
                )
        else:
            match = _N_PROBES_RE.match(name)
            if match is None:
                raise HarnessError(
                    f"unknown scenario {name!r}; valid: baseline, probed-0, inline-N, async-N, inline-N-mlp, "
                    "async-N-mlp, mixed, async-slow, async-slow:<policy>"
                )
            mode, count, mlp = match.group(1), int(match.group(2)), match.group(3)
            if count < 1:
                raise HarnessError(f"scenario {name!r}: use probed-0 for zero probes")
            impl = "mlp" if mlp else "norm"
            units.append(Scenario(name, probed=True, slots=tuple(ProbeSlot(impl, mode) for _ in range(count))))
    names = [u.name for u in units]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise HarnessError(f"scenario(s) listed twice: {', '.join(duplicates)}")
    if not units:
        raise HarnessError("--scenarios is empty")
    return units


def layer_for(index: int, count: int, num_layers: int) -> int:
    """Spread `count` extraction points evenly over the model's layers."""
    return int((index + 0.5) * num_layers / count) % num_layers


def build_spec(scenario: Scenario, num_layers: int) -> dict[str, Any]:
    """The probe spec (as a dict, validated by `ProbedModel`) for `scenario`."""
    points = []
    for i, slot in enumerate(scenario.slots):
        point: dict[str, Any] = {
            "name": f"bench_{i}_{slot.impl}_{slot.mode}",
            "layers": layer_for(i, len(scenario.slots), num_layers),
            "tensor_type": "residual_stream",
            "position": "generated[*]",
            "probe_type": f"bench_{slot.impl}",
            # trajectory: async execution requires it; inline uses it too so
            # inline and async scenarios run the same probe code.
            "probe_kind": "trajectory",
            "execution_mode": slot.mode,
            "probe_args": {"inline": slot.mode == "inline"},
        }
        if slot.mode == "async" and scenario.queue_depth is not None:
            point["queue_depth"] = scenario.queue_depth
        points.append(point)
    return {"version": "1", "extraction_points": points}


# ---------------------------------------------------------------------------
# Probes and metrics collection
# ---------------------------------------------------------------------------


class ProbeStats:
    """Thread-safe totals of the benchmark probes' own work for one repeat."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.activations = 0
            self.compute_s = 0.0
            self.inline_compute_s = 0.0

    def add(self, seconds: float, inline: bool) -> None:
        with self._lock:
            self.activations += 1
            self.compute_s += seconds
            if inline:
                self.inline_compute_s += seconds


class MLPWeights:
    """Random two-layer MLP weights shared by every MLP probe instance of a
    run (built lazily on the first activation's device and dtype), so a probe
    spawn never pays for allocating them."""

    def __init__(self, seed: int, hidden: int = MLP_HIDDEN) -> None:
        self._seed = seed
        self._hidden = hidden
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, str, str], tuple[Any, Any]] = {}

    def get(self, dim: int, device: Any, dtype: Any) -> tuple[Any, Any]:
        key = (dim, str(device), str(dtype))
        with self._lock:
            weights = self._cache.get(key)
            if weights is None:
                import torch

                gen = torch.Generator().manual_seed(self._seed)
                w1 = torch.randn(dim, self._hidden, generator=gen) / math.sqrt(dim)
                w2 = torch.randn(self._hidden, 1, generator=gen) / math.sqrt(self._hidden)
                weights = (w1.to(device=device, dtype=dtype), w2.to(device=device, dtype=dtype))
                self._cache[key] = weights
            return weights


def _make_probe_classes() -> dict[str, type]:
    """The benchmark probe classes, built lazily so `--help` and argument
    errors don't import torch or undercurrent."""
    import torch

    from undercurrent.core import Probe, ProbeResult

    class _BenchProbe(Probe):
        """Trajectory probe that scores every activation, never intervenes,
        and times its own work (`ProbeStats`)."""

        probe_kind = "trajectory"

        def __init__(self, stats: ProbeStats, inline: bool = False) -> None:
            super().__init__()
            self._stats = stats
            self._inline = inline
            self._count = 0
            self._max = -math.inf

        def on_start(self, request_ctx: Any) -> None:
            pass

        def score(self, tensor: Any) -> float:
            raise NotImplementedError

        def on_activation(self, record: Any) -> None:
            start = time.perf_counter()
            value = self.score(record.tensor)
            self._stats.add(time.perf_counter() - start, self._inline)
            self._count += 1
            self._max = max(self._max, value)
            return None  # never intervenes: the benchmark measures cost, not decisions

        def on_end(self, request_ctx: Any) -> Any:
            return ProbeResult(
                request_ctx.request_id,
                self.extraction_point_name,
                verdict={"activations": self._count, "max_score": self._max if self._count else None},
            )

    class NormProbe(_BenchProbe):
        """The cheapest realistic probe: one L2 norm and a host sync."""

        def score(self, tensor: Any) -> float:
            return float(torch.linalg.vector_norm(tensor.float()))

    class MLPProbe(_BenchProbe):
        """A small random-weight MLP head (d -> 256 -> 1, sigmoid): roughly
        what a trained linear/MLP classifier probe costs."""

        def __init__(self, stats: ProbeStats, weights: MLPWeights, inline: bool = False) -> None:
            super().__init__(stats, inline)
            self._weights = weights

        def score(self, tensor: Any) -> float:
            x = tensor.reshape(-1, tensor.shape[-1])
            w1, w2 = self._weights.get(x.shape[-1], x.device, x.dtype)
            return float(torch.sigmoid(torch.relu(x @ w1) @ w2).max())

    class SleepProbe(_BenchProbe):
        """A deliberately slow observer: sleeps `delay_s` per activation."""

        def __init__(self, stats: ProbeStats, delay_s: float, inline: bool = False) -> None:
            super().__init__(stats, inline)
            self._delay_s = delay_s

        def score(self, tensor: Any) -> float:
            time.sleep(self._delay_s)
            return 0.0

    return {"norm": NormProbe, "mlp": MLPProbe, "sleep": SleepProbe}


def _make_metrics_sink() -> Any:
    from undercurrent.router.metrics import MetricsSink

    class BenchMetricsSink(MetricsSink):
        """Collects drops, errors and the total async backlog over time.

        Queue depth is reported per (request, extraction point); the series
        records the sum over all live queues after every change, i.e. the
        total async backlog.
        """

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.reset()

        def reset(self) -> None:
            with self._lock:
                self.t0 = time.perf_counter()
                self.drops = 0
                self.errors = 0
                self.async_activations = 0
                self._depths: dict[tuple[str, str], int] = {}
                self.series: list[tuple[float, int]] = []

        def record_queue_depth(self, request_id: str, extraction_point_name: str, depth: int) -> None:
            now = time.perf_counter()
            with self._lock:
                key = (request_id, extraction_point_name)
                if depth:
                    self._depths[key] = depth
                else:
                    self._depths.pop(key, None)
                self.series.append((now - self.t0, sum(self._depths.values())))

        def record_drop(self, request_id: str, extraction_point_name: str) -> None:
            with self._lock:
                self.drops += 1

        def record_activation(self, request_id: str, extraction_point_name: str, latency_seconds: float) -> None:
            with self._lock:
                self.async_activations += 1

        def record_probe_error(self, request_id: str, extraction_point_name: str) -> None:
            with self._lock:
                self.errors += 1

    return BenchMetricsSink()


def downsample_series(
    series: Sequence[tuple[float, int]], points: int = QUEUE_DEPTH_SERIES_POINTS
) -> list[list[float]]:
    """At most `points` `[t_s, depth]` pairs: the series is cut into equal
    time buckets and each bucket keeps its maximum depth (so peaks survive)."""
    if not series:
        return []
    if len(series) <= points:
        return [[round(t, 6), d] for t, d in series]
    end = series[-1][0] or 1e-9
    buckets: dict[int, tuple[float, int]] = {}
    for t, d in series:
        b = min(int(t / end * points), points - 1)
        if b not in buckets or d > buckets[b][1]:
            buckets[b] = (t, d)
    return [[round(t, 6), d] for t, d in (buckets[b] for b in sorted(buckets))]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Neutral filler text the prompts are cut from. Its content is irrelevant:
# only the token count matters, and every scenario gets the same prompts.
_PROMPT_TEXT = (
    "A river looks calm from the bank, but the water beneath the surface keeps moving in its own "
    "direction. Engineers who build bridges study these currents for years before they pour the "
    "first foundation. They measure the speed of the flow at different depths, the shape of the "
    "riverbed, the way sediment settles after a flood and the seasons when the level rises. "
    "Each measurement is written down, compared with the last one and checked again by a second "
    "team, because a small error in the model of the current becomes a large error in the "
    "structure that has to stand in it. "
)


def make_prompts(tokenizer: Any, num_prompts: int, input_len: int, seed: int) -> list[str]:
    """`num_prompts` deterministic prompts of about `input_len` tokens each,
    cut from `_PROMPT_TEXT` at seeded offsets."""
    ids = tokenizer(_PROMPT_TEXT, add_special_tokens=False)["input_ids"]
    if not ids:
        raise HarnessError("the tokenizer produced no tokens for the prompt text")
    rng = random.Random(seed)
    prompts = []
    for _ in range(num_prompts):
        start = rng.randrange(len(ids))
        window = [ids[(start + j) % len(ids)] for j in range(input_len)]
        prompts.append(tokenizer.decode(window, skip_special_tokens=True))
    return prompts


def prompts_digest(prompts: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(prompts)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class BatchResult:
    latency_s: float
    output_tokens: int
    ttft_s: list[float] = field(default_factory=list)
    ttlt_s: list[float] = field(default_factory=list)


class Session:
    """A loaded scenario: baseline or a `ProbedModel` with its probes."""

    def run_batch(self, prompts: list[str]) -> BatchResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _transformers_dtype_kwargs(dtype: str) -> dict[str, Any]:
    if dtype == "auto":
        return {"dtype": "auto"} if _transformers_uses_dtype_kwarg() else {"torch_dtype": "auto"}
    import torch

    value = getattr(torch, dtype)
    return {"dtype": value} if _transformers_uses_dtype_kwarg() else {"torch_dtype": value}


def _transformers_uses_dtype_kwarg() -> bool:
    """transformers 4.56 renamed `from_pretrained(torch_dtype=)` to `dtype=`."""
    from packaging.version import Version

    return Version(importlib.metadata.version("transformers")) >= Version("4.56")


class HFRunner:
    """Loads the transformers model once; every scenario reuses it.

    `baseline` calls `model.generate` directly. Probed scenarios wrap the
    same model object in a fresh `ProbedModel`, whose `close()` removes the
    adapter's hooks again, so later scenarios start from the plain model.
    """

    timing_supported = True

    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.device = args.device
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, **_transformers_dtype_kwargs(args.dtype))
        self.model.to(self.device).eval()
        self.num_layers = _num_layers(AutoConfig.from_pretrained(args.model))
        pad = self.tokenizer.pad_token_id
        self.pad_token_id = pad if pad is not None else self.tokenizer.eos_token_id

    def _streamer(self) -> Any:
        from transformers.generation.streamers import BaseStreamer

        class TokenTimer(BaseStreamer):
            """Timestamps every generated token. `generate` first passes the
            prompt ids, then one put() per new token."""

            def __init__(self) -> None:
                self.start = time.perf_counter()
                self.first: float | None = None
                self.last: float | None = None
                self.tokens = 0
                self._seen_prompt = False

            def put(self, value: Any) -> None:
                if not self._seen_prompt:
                    self._seen_prompt = True
                    return
                now = time.perf_counter()
                if self.first is None:
                    self.first = now
                self.last = now
                self.tokens += int(value.numel())

            def end(self) -> None:
                pass

        return TokenTimer()

    def _timed(self, prompts: list[str], generate_one: Callable[[str, Any], None]) -> BatchResult:
        # HF generates one prompt at a time (ProbedModel's HF backend does
        # too), so a batch is its prompts back to back, identically for
        # baseline and probed scenarios.
        start = time.perf_counter()
        ttft, ttlt, tokens = [], [], 0
        for prompt in prompts:
            timer = self._streamer()
            generate_one(prompt, timer)
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize()
            tokens += timer.tokens
            if timer.first is not None and timer.last is not None:
                ttft.append(timer.first - timer.start)
                ttlt.append(timer.last - timer.start)
        return BatchResult(time.perf_counter() - start, tokens, ttft, ttlt)

    def open(self, scenario: Scenario, probes: dict[str, Any], router_kwargs: dict[str, Any], sink: Any) -> Session:
        runner = self
        n = self.args.max_new_tokens

        if not scenario.probed:

            class Baseline(Session):
                def run_batch(self, prompts: list[str]) -> BatchResult:
                    def one(prompt: str, timer: Any) -> None:
                        inputs = runner.tokenizer(prompt, return_tensors="pt").to(runner.device)
                        with runner.torch.no_grad():
                            runner.model.generate(
                                **inputs,
                                max_new_tokens=n,
                                min_new_tokens=n,
                                do_sample=False,
                                pad_token_id=runner.pad_token_id,
                                streamer=timer,
                            )

                    return runner._timed(prompts, one)

            return Baseline()

        from undercurrent.model import ProbedModel

        pm = ProbedModel(
            self.model,
            tokenizer=self.tokenizer,
            spec=build_spec(scenario, self.num_layers),
            probes=probes,
            device=self.device,
            router_kwargs=router_kwargs,
            metrics_sink=sink,
        )

        class Probed(Session):
            def run_batch(self, prompts: list[str]) -> BatchResult:
                def one(prompt: str, timer: Any) -> None:
                    pm.generate(prompt, max_new_tokens=n, min_new_tokens=n, temperature=0, streamer=timer)

                return runner._timed(prompts, one)

            def close(self) -> None:
                pm.close()

        return Probed()

    def reset_peak_memory(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self) -> int | None:
        if self.device.startswith("cuda"):
            return int(self.torch.cuda.max_memory_allocated(self.device))
        return None

    def close(self) -> None:
        pass


class VLLMRunner:
    """vLLM: `baseline` is a plain `vllm.LLM`; probed scenarios use
    `ProbedModel(backend="vllm")`. Each scenario builds its own engine (an
    adapter binds one Router for life), so the harness runs every vLLM
    scenario in its own subprocess (`--isolate auto`)."""

    timing_supported = False

    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoConfig, AutoTokenizer

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.num_layers = _num_layers(AutoConfig.from_pretrained(args.model))
        self.engine_kwargs = dict(args.vllm_engine_kwargs)
        if args.dtype != "auto":
            self.engine_kwargs.setdefault("dtype", args.dtype)
        self.engine_kwargs.setdefault("seed", args.seed)

    def open(self, scenario: Scenario, probes: dict[str, Any], router_kwargs: dict[str, Any], sink: Any) -> Session:
        n = self.args.max_new_tokens

        if not scenario.probed:
            from vllm import LLM, SamplingParams

            llm = LLM(model=self.args.model, **self.engine_kwargs)
            params = SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n, ignore_eos=True)

            class Baseline(Session):
                def run_batch(self, prompts: list[str]) -> BatchResult:
                    start = time.perf_counter()
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    latency = time.perf_counter() - start
                    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                    return BatchResult(latency, tokens)

                def close(self) -> None:
                    shutdown = getattr(getattr(llm, "llm_engine", None), "shutdown", None)
                    if callable(shutdown):
                        shutdown()

            return Baseline()

        from undercurrent.model import ProbedModel

        pm = ProbedModel(
            self.args.model,
            backend="vllm",
            spec=build_spec(scenario, self.num_layers),
            probes=probes,
            router_kwargs=router_kwargs,
            metrics_sink=sink,
            max_concurrency=self.args.batch_size,
            **self.engine_kwargs,
        )

        class Probed(Session):
            def run_batch(self, prompts: list[str]) -> BatchResult:
                start = time.perf_counter()
                pm.generate(prompts, max_new_tokens=n, min_new_tokens=n, temperature=0, ignore_eos=True)
                # ProbedModel returns text, not token ids; min = max tokens
                # with ignore_eos makes every output exactly n tokens.
                return BatchResult(time.perf_counter() - start, n * len(prompts))

            def close(self) -> None:
                pm.close()

        return Probed()

    def reset_peak_memory(self) -> None:
        pass

    def peak_memory(self) -> int | None:
        # The engine allocates its own memory (KV cache sized by
        # gpu_memory_utilization), partly in its own process; the harness
        # process's allocator stats would be misleading.
        return None

    def close(self) -> None:
        pass


def _num_layers(config: Any) -> int:
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        config = get_text_config()
    layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(layers, int) or layers < 1:
        raise HarnessError("could not read num_hidden_layers from the model config")
    return layers


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method); None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def run_scenario(
    runner: Any,
    scenario: Scenario,
    prompts: list[str],
    args: argparse.Namespace,
    run_id: str,
) -> list[dict[str, Any]]:
    """Warm up, then measure `args.repeats` passes over all prompts. One row per repeat."""
    import torch

    classes = _make_probe_classes()
    stats = ProbeStats()
    weights = MLPWeights(args.seed)
    from undercurrent.core import ProbeFactory

    probes = {
        "bench_norm": ProbeFactory(classes["norm"], {"stats": stats}),
        "bench_mlp": ProbeFactory(classes["mlp"], {"stats": stats, "weights": weights}),
        "bench_sleep": ProbeFactory(classes["sleep"], {"stats": stats, "delay_s": args.probe_delay_ms / 1000.0}),
    }
    router_kwargs: dict[str, Any] = {}
    if args.worker_pool_size is not None:
        router_kwargs["worker_pool_size"] = args.worker_pool_size
    if scenario.overflow_policy is not None:
        from undercurrent.router.overflow import OverflowPolicy

        router_kwargs["default_overflow_policy"] = OverflowPolicy(scenario.overflow_policy)
    sink = _make_metrics_sink() if scenario.probed else None

    from undercurrent.router.router import default_worker_pool_size

    pool_size = args.worker_pool_size if args.worker_pool_size is not None else default_worker_pool_size()
    batches = [prompts[i : i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    prompt_tokens = [len(runner.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    digest = prompts_digest(prompts)

    log.info("scenario %s: loading", scenario.name)
    session = runner.open(scenario, probes, router_kwargs, sink)
    rows = []
    try:
        for i in range(args.warmup):
            session.run_batch(batches[i % len(batches)])
        for repeat in range(args.repeats):
            stats.reset()
            if sink is not None:
                sink.reset()
            torch.manual_seed(args.seed + repeat)
            runner.reset_peak_memory()
            results = []
            start = time.perf_counter()
            for batch in batches:
                results.append(session.run_batch(batch))
            wall = time.perf_counter() - start
            latencies = [r.latency_s for r in results]
            ttft = [t for r in results for t in r.ttft_s]
            ttlt = [t for r in results for t in r.ttlt_s]
            tokens = sum(r.output_tokens for r in results)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "backend": args.backend,
                    "model": args.model,
                    "scenario": scenario.name,
                    "repeat": repeat,
                    "num_probes": len(scenario.slots),
                    "num_inline_probes": scenario.num_inline,
                    "num_async_probes": scenario.num_async,
                    "probe_impls": scenario.probe_impls,
                    "overflow_policy": scenario.overflow_policy,
                    "queue_depth": scenario.queue_depth,
                    "probe_delay_ms": args.probe_delay_ms if scenario.is_slow else None,
                    "worker_pool_size": pool_size if scenario.probed else None,
                    "num_prompts": len(prompts),
                    "batch_size": args.batch_size,
                    "max_new_tokens": args.max_new_tokens,
                    "input_len": args.input_len,
                    "prompt_tokens_mean": _mean(prompt_tokens),
                    "prompts_sha256": digest,
                    "seed": args.seed,
                    "num_batches": len(batches),
                    "wall_time_s": wall,
                    "batch_latencies_s": latencies,
                    "latency_p50_s": percentile(latencies, 50),
                    "latency_p90_s": percentile(latencies, 90),
                    "latency_p99_s": percentile(latencies, 99),
                    "latency_mean_s": _mean(latencies),
                    "output_tokens": tokens,
                    "output_tokens_counted": bool(runner.timing_supported or not scenario.probed),
                    "output_tokens_per_s": tokens / wall if wall > 0 else None,
                    "ttft_s": ttft if runner.timing_supported else None,
                    "ttft_p50_s": percentile(ttft, 50),
                    "ttft_p90_s": percentile(ttft, 90),
                    "ttft_p99_s": percentile(ttft, 99),
                    "ttlt_s": ttlt if runner.timing_supported else None,
                    "ttlt_p50_s": percentile(ttlt, 50),
                    "peak_gpu_mem_bytes": runner.peak_memory(),
                    "probe_activations": stats.activations,
                    "probe_compute_s": stats.compute_s,
                    "inline_probe_compute_s": stats.inline_compute_s,
                    "queue_drops": sink.drops if sink is not None else None,
                    "probe_errors": sink.errors if sink is not None else None,
                    "queue_depth_max": max((d for _, d in sink.series), default=0) if sink is not None else None,
                    "queue_depth_series": downsample_series(sink.series) if sink is not None else None,
                    # Filled in by add_overhead() once the baseline is known.
                    "overhead_pct_latency_p50": None,
                    "overhead_pct_tokens_per_s": None,
                    "overhead_pct_ex_inline_probe_compute": None,
                    "latency_flat": None,
                }
            )
            log.info(
                "scenario %s: repeat %d/%d: wall %.3fs, %d tokens",
                scenario.name,
                repeat + 1,
                args.repeats,
                wall,
                tokens,
            )
    finally:
        session.close()
    return rows


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = (
    "schema_version",
    "backend",
    "model",
    "scenario",
    "repeats",
    "num_probes",
    "num_inline_probes",
    "num_async_probes",
    "probe_impls",
    "overflow_policy",
    "queue_depth",
    "probe_delay_ms",
    "worker_pool_size",
    "num_prompts",
    "batch_size",
    "max_new_tokens",
    "input_len",
    "latency_p50_s",
    "latency_p90_s",
    "latency_p99_s",
    "latency_mean_s",
    "output_tokens_per_s_mean",
    "output_tokens_per_s_std",
    "ttft_p50_s",
    "ttft_p90_s",
    "ttft_p99_s",
    "ttlt_p50_s",
    "peak_gpu_mem_bytes",
    "probe_activations_mean",
    "probe_compute_s_mean",
    "inline_probe_compute_s_mean",
    "queue_drops_total",
    "probe_errors_total",
    "queue_depth_max",
    "overhead_pct_latency_p50",
    "overhead_pct_tokens_per_s",
    "overhead_pct_ex_inline_probe_compute",
    "latency_flat",
)


def _pct(value: float | None, reference: float | None) -> float | None:
    if value is None or not reference:
        return None
    return (value / reference - 1.0) * 100.0


def _sum_or_none(values: Sequence[Any]) -> Any:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _max_or_none(values: Sequence[Any]) -> Any:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per scenario (in first-seen order), pooling all repeats."""
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)
    out = []
    for name, group in by_scenario.items():
        first = group[0]
        latencies = [x for r in group for x in r["batch_latencies_s"]]
        ttft = [x for r in group for x in (r["ttft_s"] or [])]
        ttlt = [x for r in group for x in (r["ttlt_s"] or [])]
        tps = [r["output_tokens_per_s"] for r in group if r["output_tokens_per_s"] is not None]
        summary = {key: first[key] for key in SUMMARY_COLUMNS if key in first}
        summary.update(
            {
                "scenario": name,
                "repeats": len(group),
                "latency_p50_s": percentile(latencies, 50),
                "latency_p90_s": percentile(latencies, 90),
                "latency_p99_s": percentile(latencies, 99),
                "latency_mean_s": _mean(latencies),
                "output_tokens_per_s_mean": _mean(tps),
                "output_tokens_per_s_std": statistics.stdev(tps) if len(tps) > 1 else None,
                "ttft_p50_s": percentile(ttft, 50),
                "ttft_p90_s": percentile(ttft, 90),
                "ttft_p99_s": percentile(ttft, 99),
                "ttlt_p50_s": percentile(ttlt, 50),
                "peak_gpu_mem_bytes": _max_or_none([r["peak_gpu_mem_bytes"] for r in group]),
                "probe_activations_mean": _mean([r["probe_activations"] for r in group]),
                "probe_compute_s_mean": _mean([r["probe_compute_s"] for r in group]),
                "inline_probe_compute_s_mean": _mean([r["inline_probe_compute_s"] for r in group]),
                "queue_drops_total": _sum_or_none([r["queue_drops"] for r in group]),
                "probe_errors_total": _sum_or_none([r["probe_errors"] for r in group]),
                "queue_depth_max": _max_or_none([r["queue_depth_max"] for r in group]),
                "_wall_time_s_mean": _mean([r["wall_time_s"] for r in group]),
            }
        )
        out.append(summary)
    return out


def add_overhead(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], flat_tolerance_pct: float) -> None:
    """Fill the overhead fields of every row and summary, relative to the
    `baseline` scenario's summary. Left None when there is no baseline."""
    base = next((s for s in summaries if s["scenario"] == "baseline"), None)
    if base is None:
        for summary in summaries:
            summary.setdefault("overhead_pct_latency_p50", None)
            summary.setdefault("overhead_pct_tokens_per_s", None)
            summary.setdefault("overhead_pct_ex_inline_probe_compute", None)
            summary.setdefault("latency_flat", None)
        return
    base_p50 = base["latency_p50_s"]
    base_tps = base["output_tokens_per_s_mean"]
    base_wall = base["_wall_time_s_mean"]
    base_gen = base["ttlt_p50_s"] if base["ttlt_p50_s"] is not None else base_p50

    def tps_overhead(tps: float | None) -> float | None:
        if tps is None or not base_tps:
            return None
        return (1.0 - tps / base_tps) * 100.0

    def flat(scenario: str, gen_p50: float | None) -> bool | None:
        if not scenario.startswith("async-slow:"):
            return None
        overhead = _pct(gen_p50, base_gen)
        return None if overhead is None else overhead <= flat_tolerance_pct

    for row in rows:
        row["overhead_pct_latency_p50"] = _pct(row["latency_p50_s"], base_p50)
        row["overhead_pct_tokens_per_s"] = tps_overhead(row["output_tokens_per_s"])
        row["overhead_pct_ex_inline_probe_compute"] = _pct(
            row["wall_time_s"] - row["inline_probe_compute_s"], base_wall
        )
        gen = row["ttlt_p50_s"] if row["ttlt_p50_s"] is not None else row["latency_p50_s"]
        row["latency_flat"] = flat(row["scenario"], gen)
    for summary in summaries:
        summary["overhead_pct_latency_p50"] = _pct(summary["latency_p50_s"], base_p50)
        summary["overhead_pct_tokens_per_s"] = tps_overhead(summary["output_tokens_per_s_mean"])
        wall = summary["_wall_time_s_mean"]
        inline = summary["inline_probe_compute_s_mean"] or 0.0
        summary["overhead_pct_ex_inline_probe_compute"] = _pct(None if wall is None else wall - inline, base_wall)
        gen = summary["ttlt_p50_s"] if summary["ttlt_p50_s"] is not None else summary["latency_p50_s"]
        summary["latency_flat"] = flat(summary["scenario"], gen)


def write_outputs(
    run_dir: Path, env: dict[str, Any], rows: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({k: _csv_value(summary.get(k)) for k in SUMMARY_COLUMNS})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return value


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _cpu_model() -> str | None:
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def collect_env(args: argparse.Namespace, argv: list[str], run_id: str, scenarios: list[Scenario]) -> dict[str, Any]:
    import torch

    repo = Path(__file__).resolve().parents[1]
    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": props.name, "total_memory_bytes": int(props.total_memory)})
    driver = None
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        driver = out.splitlines()[0].strip() if out else None
    import undercurrent

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "cli_argv": argv,
        "args": {k: v for k, v in sorted(vars(args).items()) if k != "worker_output"},
        "scenarios": [s.name for s in scenarios],
        "dry_run": bool(args.dry_run),
        "device": args.device,
        "git_sha": sha,
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": {"model": _cpu_model(), "logical_cores": os.cpu_count()},
        "gpu": {
            "count": len(gpus),
            "devices": gpus,
            "driver_version": driver,
            "cuda_version": torch.version.cuda,
        },
        "versions": {
            "undercurrent": getattr(undercurrent, "__version__", None),
            "torch": torch.__version__,
            "transformers": _pkg_version("transformers"),
            "vllm": _pkg_version("vllm"),
            "numpy": _pkg_version("numpy"),
        },
        "failed_scenarios": [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _engine_kwarg(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    key, raw = text.split("=", 1)
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmarks/run.py",
        description="Measure Undercurrent's probing overhead against plain generation. See benchmarks/README.md.",
    )
    p.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    p.add_argument("--model", help=f"HF hub id or local path (required unless --dry-run, which uses {DRY_RUN_MODEL})")
    p.add_argument("--num-prompts", type=int, help=f"prompts per repeat (default {_DEFAULT_SIZES['num_prompts']})")
    p.add_argument(
        "--batch-size",
        type=int,
        help=f"prompts per generate() call (hf: must be 1; vllm default {DEFAULT_VLLM_BATCH_SIZE})",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        help=f"tokens generated per prompt, exactly (default {_DEFAULT_SIZES['max_new_tokens']})",
    )
    p.add_argument("--input-len", type=int, help=f"prompt length in tokens (default {_DEFAULT_SIZES['input_len']})")
    p.add_argument(
        "--scenarios",
        help=f"comma-separated (default {DEFAULT_SCENARIOS}; --dry-run default: every scenario). "
        "Also: probed-0, inline-N-mlp, async-N-mlp, async-slow, async-slow:<policy>",
    )
    p.add_argument("--warmup", type=int, help=f"untimed batches per scenario (default {_DEFAULT_SIZES['warmup']})")
    p.add_argument("--repeats", type=int, help=f"timed passes per scenario (default {_DEFAULT_SIZES['repeats']})")
    p.add_argument("--out", default="benchmarks/results", help="output root (default benchmarks/results)")
    p.add_argument("--dry-run", action="store_true", help=f"CPU, {DRY_RUN_MODEL}, tiny sizes, every scenario")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--probe-delay-ms",
        type=float,
        help=f"async-slow: sleep per activation (default {_DEFAULT_SIZES['probe_delay_ms']})",
    )
    p.add_argument(
        "--slow-queue-depth",
        type=int,
        help=f"async-slow: the probe's queue_depth (default {_DEFAULT_SIZES['slow_queue_depth']})",
    )
    p.add_argument(
        "--flat-tolerance-pct",
        type=float,
        default=DEFAULT_FLAT_TOLERANCE_PCT,
        help="async-slow: latency_flat is true when generation latency is within this %% of baseline (default 10)",
    )
    p.add_argument(
        "--worker-pool-size",
        type=int,
        help="Router(worker_pool_size=...) for probed scenarios (default: the router's own default). Each live "
        "async binding pins one pool thread, so with --backend vllm use at least batch size x async probes",
    )
    p.add_argument("--device", help="hf: torch device (default cuda if available, else cpu)")
    p.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    p.add_argument(
        "--vllm-arg",
        dest="vllm_engine_kwargs",
        type=_engine_kwarg,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="vLLM engine arg for baseline and probed engines alike, e.g. gpu_memory_utilization=0.85 "
        "(VALUE is parsed as JSON when possible; repeatable)",
    )
    p.add_argument(
        "--isolate",
        choices=("auto", "process", "none"),
        default="auto",
        help="run each scenario in its own subprocess (auto: yes for vllm, no for hf)",
    )
    p.add_argument("--worker-output", help=argparse.SUPPRESS)
    return p


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply --dry-run and backend defaults; validate sizes."""
    if args.dry_run:
        if args.backend != "hf":
            raise HarnessError("--dry-run is CPU-only and supports --backend hf only")
        args.model = args.model or DRY_RUN_MODEL
        args.device = "cpu"
        for key, value in _DRY_RUN_SIZES.items():
            setattr(args, key, value)
        args.scenarios = args.scenarios or DRY_RUN_SCENARIOS
    else:
        if not args.model:
            raise HarnessError("--model is required (or use --dry-run)")
        for key, value in _DEFAULT_SIZES.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        args.scenarios = args.scenarios or DEFAULT_SCENARIOS
    if args.batch_size is None:
        args.batch_size = 1 if args.backend == "hf" else DEFAULT_VLLM_BATCH_SIZE
    if args.backend == "hf":
        if args.batch_size != 1:
            raise HarnessError(
                "--backend hf generates one prompt at a time (ProbedModel's HF backend has max_concurrency=1), "
                "so --batch-size must be 1; use --backend vllm for batched serving"
            )
        if args.vllm_engine_kwargs:
            raise HarnessError("--vllm-arg only applies to --backend vllm")
        if args.device is None:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device is not None:
        raise HarnessError("--device only applies to --backend hf; vLLM places the model itself")
    for key in ("num_prompts", "batch_size", "max_new_tokens", "input_len", "repeats", "slow_queue_depth"):
        if getattr(args, key) < 1:
            raise HarnessError(f"--{key.replace('_', '-')} must be >= 1")
    if args.worker_pool_size is not None and args.worker_pool_size < 1:
        raise HarnessError("--worker-pool-size must be >= 1")
    if args.warmup < 0 or args.probe_delay_ms < 0:
        raise HarnessError("--warmup and --probe-delay-ms must be >= 0")
    if args.isolate == "auto":
        args.isolate = "process" if args.backend == "vllm" else "none"
    args.vllm_engine_kwargs = dict(args.vllm_engine_kwargs)
    return args


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_vllm_environment() -> None:
    """Fail fast, with one actionable line, if vLLM can't run here."""
    if not _module_available("vllm"):
        raise HarnessError(
            "--backend vllm needs vLLM, which undercurrent never installs by default. Run the harness in your "
            'vLLM environment or image, or `pip install "undercurrent[vllm]"`.'
        )
    import torch

    if not torch.cuda.is_available():
        raise HarnessError(
            "--backend vllm needs a CUDA GPU, and torch.cuda.is_available() is False here. Run on a GPU host, "
            "or use --backend hf (e.g. --dry-run) on CPU."
        )
    from undercurrent.adapters.vllm.version_check import check_vllm_version

    try:
        check_vllm_version()
    except Exception as exc:  # noqa: BLE001 -- surfaced as a one-line harness error
        raise HarnessError(str(exc)) from exc


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model.strip("/").replace("/", "--")).strip("-")
    return slug or "model"


def _child_argv(args: argparse.Namespace, scenario: Scenario, output: str) -> list[str]:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend", args.backend,
        "--model", args.model,
        "--num-prompts", str(args.num_prompts),
        "--batch-size", str(args.batch_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--input-len", str(args.input_len),
        "--scenarios", scenario.name,
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--probe-delay-ms", str(args.probe_delay_ms),
        "--slow-queue-depth", str(args.slow_queue_depth),
        "--dtype", args.dtype,
        "--isolate", "none",
        "--worker-output", output,
    ]  # fmt: skip
    if args.device is not None:
        argv += ["--device", args.device]
    if args.worker_pool_size is not None:
        argv += ["--worker-pool-size", str(args.worker_pool_size)]
    for key, value in args.vllm_engine_kwargs.items():
        argv += ["--vllm-arg", f"{key}={json.dumps(value)}"]
    return argv


def _run_isolated(args: argparse.Namespace, scenario: Scenario) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="undercurrent-bench-") as tmp:
        output = os.path.join(tmp, "rows.json")
        proc = subprocess.run(_child_argv(args, scenario, output), check=False)
        if proc.returncode != 0 or not os.path.exists(output):
            raise RuntimeError(f"scenario subprocess exited with code {proc.returncode}")
        with open(output, encoding="utf-8") as fh:
            rows: list[dict[str, Any]] = json.load(fh)
        return rows


def _make_runner(args: argparse.Namespace) -> Any:
    return HFRunner(args) if args.backend == "hf" else VLLMRunner(args)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="[bench] %(message)s", stream=sys.stderr)
    log.setLevel(logging.INFO)
    try:
        return _main(args, argv)
    except HarnessError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2


def _main(args: argparse.Namespace, argv: list[str]) -> int:
    args = resolve_args(args)
    scenarios = parse_scenarios(args.scenarios, slow_queue_depth=args.slow_queue_depth)
    if args.backend == "vllm":
        check_vllm_environment()

    if args.worker_output:  # a scenario subprocess of an --isolate process run
        runner = _make_runner(args)
        prompts = make_prompts(runner.tokenizer, args.num_prompts, args.input_len, args.seed)
        rows = [row for s in scenarios for row in run_scenario(runner, s, prompts, args, run_id="")]
        Path(args.worker_output).write_text(json.dumps(rows), encoding="utf-8")
        return 0

    started = _dt.datetime.now(_dt.timezone.utc)
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{args.backend}-{model_slug(args.model)}"
    run_dir = Path(args.out) / run_id
    env = collect_env(args, argv, run_id, scenarios)

    runner = None if args.isolate == "process" else _make_runner(args)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        try:
            if runner is None:
                new_rows = _run_isolated(args, scenario)
            else:
                prompts = make_prompts(runner.tokenizer, args.num_prompts, args.input_len, args.seed)
                new_rows = run_scenario(runner, scenario, prompts, args, run_id)
        except Exception as exc:  # noqa: BLE001 -- record, keep measuring the other scenarios
            log.error("scenario %s failed:\n%s", scenario.name, traceback.format_exc())
            env["failed_scenarios"].append({"scenario": scenario.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for row in new_rows:
            row["run_id"] = run_id
        rows.extend(new_rows)
    if runner is not None:
        runner.close()

    digests = sorted({row["prompts_sha256"] for row in rows})
    if len(digests) > 1:
        raise HarnessError(f"scenarios saw different prompts ({len(digests)} distinct digests); results discarded")
    summaries = summarize(rows)
    add_overhead(rows, summaries, args.flat_tolerance_pct)
    write_outputs(run_dir, env, rows, summaries)
    print(run_dir)
    if env["failed_scenarios"]:
        names = ", ".join(f["scenario"] for f in env["failed_scenarios"])
        print(f"{build_parser().prog}: error: scenario(s) failed: {names} (see env.json)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

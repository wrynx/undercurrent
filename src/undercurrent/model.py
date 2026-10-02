"""ProbedModel: the high-level front door to Undercurrent.

Load a model, attach probes from a spec, generate, read the results::

    from undercurrent import ProbedModel

    with ProbedModel.from_pretrained("gpt2", spec="probes.yaml") as m:
        out = m.generate("Hello", max_new_tokens=32)
        print(out.text, out.aborted, out.probe_results)

`ProbedModel` hides the spec parsing, the probe registry, the `Router`'s
lifecycle and the engine adapter; it removes none of them. Everything stays
reachable for advanced use:

- `log_sink=` / `metrics_sink=` attach observation and metrics sinks
  (`Router.attach_log_sink` / `Router.attach_metrics_sink`).
- `router_kwargs=` is forwarded to `Router(...)` (worker pool size, queue
  depth, overflow policy, drain timeout, default intervention policy, ...).
- `router=` brings your own `Router`, which the caller keeps owning.
- `backend=` takes an `EngineAdapter` instance as well as a backend name.
- `m.router` and `m.adapter` expose the live objects.

Backends
--------
A backend (`_Backend`) knows how to build its engine adapter, which
`tensor_type`s it supports, how many layers the model has, how to translate
the normalised generation kwargs (`max_new_tokens`, `temperature`, `top_p`,
`seed`, `stop`) and how many `generate()` calls may run at once. This module
ships the Hugging Face transformers backend (`"hf"`, one generation at a
time) and the vLLM backend (`"vllm"`, concurrent generations batched by
vLLM's scheduler). torch, transformers and vllm are imported lazily, inside
the backends, so `import undercurrent.model` stays light.

When a backend allows more than one generation at a time, `generate()` on a
list of prompts runs them on a thread pool of `min(len(prompts),
max_concurrency)` workers and returns the outputs in prompt order.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Literal, overload

from .adapters.base import EngineAdapter
from .core import Probe, ProbeAction, ProbeFactory, ProbeResult, ProbeSignal
from .core.registry import ProbeNotFoundError, ProbeRegistry
from .errors import ProbingError, ProbingRuntimeError, ProbingTypeError, ProbingValueError, did_you_mean
from .router import MetricsSink, Router, RouterError
from .router.binding import SupportsLogSink
from .spec import (
    ExecutionMode,
    ExtractionPoint,
    ProbeSpec,
    TensorType,
    load_spec,
)

logger = logging.getLogger("undercurrent")

SpecInput = str | os.PathLike[str] | Mapping[str, Any] | ProbeSpec | Iterable[ExtractionPoint] | None
"""Everything `ProbedModel(spec=...)` accepts: a YAML file path, a YAML
string, a dict, a parsed `ProbeSpec`, a list of `ExtractionPoint`, or None
(no probes)."""

OnResult = Callable[["GenerationOutput"], None]

DEFAULT_MAX_NEW_TOKENS = 32
"""Used when the generation length isn't given (HF: neither `max_new_tokens`
nor `max_length`; vLLM: neither `max_new_tokens` nor `max_tokens`)."""

DEFAULT_VLLM_MAX_CONCURRENCY = 64
"""How many `generate()` calls the vLLM backend runs at once by default."""

_TEXT_PREVIEW_CHARS = 60


class ProbedModelConfigError(ProbingError, ValueError):
    """The spec, probes or backend given to `ProbedModel` don't fit together.

    Raised at construction time (fail fast), before any generation. The
    message names the extraction point and says how to fix it.
    """


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationOutput:
    """The result of one `ProbedModel.generate()` call for one prompt.

    Attributes:
        text: the generated text (the prompt not included).
        prompt: the prompt it was generated from.
        request_id: the id the probes saw for this generation.
        probe_results: `{extraction_point_name: ProbeResult}` for every
            extraction point in the spec, async ones included.
        aborted: whether an inline probe stopped generation early.
        abort_reason: human-readable reason when `aborted`, else None.
        abort_signal: the `ProbeSignal` that stopped generation, if known.
        abort_point: name of the extraction point that stopped generation.
    """

    text: str
    prompt: str
    request_id: str
    probe_results: Mapping[str, ProbeResult] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str | None = None
    abort_signal: ProbeSignal | None = None
    abort_point: str | None = None

    def __repr__(self) -> str:
        text = self.text
        if len(text) > _TEXT_PREVIEW_CHARS:
            text = text[: _TEXT_PREVIEW_CHARS - 1] + "…"
        results = ", ".join(
            f"{name}: verdict={_short_repr(result.verdict)}" for name, result in self.probe_results.items()
        )
        parts = [f"text={text!r}"]
        if self.aborted:
            parts.append(f"aborted=True, abort_point={self.abort_point!r}")
        parts.append(f"probe_results={{{results}}}")
        return f"GenerationOutput({', '.join(parts)})"


def _short_repr(value: Any, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _abort_reason(point: str | None, signal: ProbeSignal | None) -> str:
    who = f"extraction point {point!r}" if point is not None else "an inline probe"
    details = []
    if signal is not None and signal.metadata.get("intervention_fallback"):
        details.append(f"intervention fallback after {signal.metadata.get('reason', 'failure')}")
    if signal is not None and signal.confidence is not None:
        details.append(f"confidence {signal.confidence:.2f}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{who} aborted generation{suffix}"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _Backend(ABC):
    """What `ProbedModel` needs to know about one inference engine."""

    #: Short name used in error messages ("hf", "vllm", ...).
    name: str = "custom"
    #: How many `generate()` calls the adapter may run at once.
    max_concurrency: int = 1
    #: Upper bound for `max_concurrency`; None means no backend-imposed limit.
    concurrency_limit: int | None = None

    def set_max_concurrency(self, value: int) -> None:
        """Override `max_concurrency` (`ProbedModel(max_concurrency=...)`)."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ProbingValueError(
                f"max_concurrency must be a positive integer (got {value!r}), e.g. max_concurrency=8"
            )
        if self.concurrency_limit is not None and value > self.concurrency_limit:
            raise ProbedModelConfigError(
                f"max_concurrency={value} is not supported by the {self.name!r} backend, which runs at most "
                f"{self.concurrency_limit} generation(s) at a time. Drop max_concurrency=, or use "
                "backend='vllm' for concurrent generation."
            )
        self.max_concurrency = value

    @abstractmethod
    def create_adapter(self, model: Any, *, device: str | None, **model_kwargs: Any) -> EngineAdapter:
        """Build the adapter and load `model` into it."""

    def supported_tensor_types(self) -> frozenset[TensorType] | None:
        """`tensor_type`s this backend can capture; None means "don't check here"."""
        return None

    def config_num_layers(self, model: Any, model_kwargs: Mapping[str, Any]) -> int | None:
        """Number of layers, read cheaply *before* the model is loaded (e.g.
        from its config), so a bad layer index fails before a slow load.
        None if unknown."""
        return None

    def num_layers(self, adapter: EngineAdapter) -> int | None:
        """Number of hookable layers in the loaded model; None if unknown."""
        return None

    def translate_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Map the normalised generation kwargs onto the engine's own. Unknown
        kwargs pass through unchanged."""
        return dict(kwargs)

    def run(
        self, adapter: EngineAdapter, request_id: str, prompt: str, kwargs: Mapping[str, Any], router: Router
    ) -> str:
        """Run one generation (extraction already registered) and return its text."""
        return adapter.generate(request_id, prompt, self.translate_kwargs(kwargs), router)

    def abort_info(self, adapter: EngineAdapter, request_id: str) -> tuple[str | None, ProbeSignal] | None:
        """`(extraction_point_name, signal)` of the abort that stopped
        `request_id`, if the adapter reports it; None otherwise."""
        last_stop = getattr(adapter, "last_stop", None)
        return last_stop(request_id) if callable(last_stop) else None

    def close(self, adapter: EngineAdapter) -> None:
        """Release what `create_adapter` set up."""
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


class _AdapterInstanceBackend(_Backend):
    """Wraps a caller-supplied `EngineAdapter`. The adapter itself enforces
    its own limits (tensor types, layers); kwargs pass through unchanged."""

    def __init__(self, adapter: EngineAdapter) -> None:
        self._adapter = adapter
        self.name = type(adapter).__name__
        self.max_concurrency = int(getattr(adapter, "max_concurrency", 1))

    def create_adapter(self, model: Any, *, device: str | None, **model_kwargs: Any) -> EngineAdapter:
        if model is not None:
            if device is not None:
                model_kwargs["device"] = device
            self._adapter.load_model(model, **model_kwargs)
        return self._adapter

    def close(self, adapter: EngineAdapter) -> None:
        pass  # the caller owns the adapter


class HFBackend(_Backend):
    """Hugging Face transformers, via `undercurrent.adapters.hf.HFEngineAdapter`.

    One generation at a time (the adapter has a single sequential decode
    loop), so a batch runs prompt by prompt. Kwarg translation:
    `temperature > 0` turns on sampling (`do_sample=True`), `temperature == 0`
    means greedy; `seed` seeds torch's RNG before generating; `stop` becomes
    `stop_strings` and the text is cut at the first stop string.
    """

    name = "hf"
    max_concurrency = 1
    concurrency_limit = 1

    def __init__(self, adapter: EngineAdapter | None = None) -> None:
        self._adapter = adapter

    def create_adapter(self, model: Any, *, device: str | None, **model_kwargs: Any) -> EngineAdapter:
        from .adapters.hf import HFEngineAdapter

        adapter = self._adapter if self._adapter is not None else HFEngineAdapter()
        if model is not None:
            adapter.load_model(model, device=device or "cpu", **model_kwargs)
        return adapter

    def supported_tensor_types(self) -> frozenset[TensorType]:
        from .adapters.hf.adapter import _UNSUPPORTED_TENSOR_TYPES

        return frozenset(t for t in TensorType if t not in _UNSUPPORTED_TENSOR_TYPES)

    def num_layers(self, adapter: EngineAdapter) -> int | None:
        return getattr(adapter, "num_layers", None)

    def translate_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(kwargs)
        out.pop("seed", None)
        stop = out.pop("stop", None)
        if stop:
            out["stop_strings"] = [stop] if isinstance(stop, str) else list(stop)
        if "max_new_tokens" not in out and "max_length" not in out:
            out["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
        temperature = out.pop("temperature", None)
        if temperature is not None:
            if temperature > 0:
                out.setdefault("do_sample", True)
                out["temperature"] = temperature
            else:
                out["do_sample"] = False
        if not out.get("do_sample", False):
            # Sampling-only knobs make transformers warn under greedy decoding.
            out.pop("top_p", None)
        return out

    def run(
        self, adapter: EngineAdapter, request_id: str, prompt: str, kwargs: Mapping[str, Any], router: Router
    ) -> str:
        hf_kwargs = self.translate_kwargs(kwargs)
        tokenizer = getattr(adapter, "tokenizer", None)
        if "stop_strings" in hf_kwargs:
            hf_kwargs.setdefault("tokenizer", tokenizer)
        if tokenizer is not None and "pad_token_id" not in hf_kwargs:
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            if pad is not None:
                hf_kwargs["pad_token_id"] = pad
        seed = kwargs.get("seed")
        if seed is not None:
            import torch

            torch.manual_seed(seed)
        text = adapter.generate(request_id, prompt, hf_kwargs, router)
        return _cut_at_stop(text, hf_kwargs.get("stop_strings"))


def _cut_at_stop(text: str, stop: Sequence[str] | None) -> str:
    """Truncate `text` before the earliest stop string (OpenAI-style: the
    stop string itself isn't returned)."""
    if not stop:
        return text
    cut = min((i for i in (text.find(s) for s in stop if s) if i >= 0), default=-1)
    return text if cut < 0 else text[:cut]


def _new_vllm_adapter() -> EngineAdapter:
    """Build an unloaded `VLLMEngineAdapter`. Raises `MissingDependencyError`
    (an `ImportError`) if vLLM isn't importable, and the adapter's own
    `VLLMAdapterLimitationError` if the installed vLLM is outside the
    supported range (see `undercurrent.adapters.vllm.version_check`).

    Module-level so tests can replace it with a fake."""
    from .adapters._optional import require_vllm

    require_vllm("backend='vllm'")  # presence check only; the adapter imports what it needs
    from .adapters.vllm import VLLMEngineAdapter

    return VLLMEngineAdapter()


class VLLMBackend(_Backend):
    """vLLM, via `undercurrent.adapters.vllm.VLLMEngineAdapter`.

    vLLM is never installed by undercurrent; it must already be in the
    environment (see docs/compatibility.md). `model` is a Hugging Face hub id
    or local path, and `**model_kwargs` are vLLM engine args
    (`gpu_memory_utilization`, `max_model_len`, `dtype`, `revision`,
    `trust_remote_code`, ...) plus the adapter's own switches
    (`allow_unsupported_executor`, ...; see `VLLMEngineAdapter.load_model`).
    vLLM places the model itself, so `device=` must be None or a CUDA device.

    Concurrency: the adapter runs vLLM's async engine on a background event
    loop, and each synchronous `generate()` blocks only its calling thread,
    so concurrent `generate()` calls on one adapter are supported and vLLM's
    scheduler batches them. `ProbedModel.generate([...])` runs up to
    `max_concurrency` prompts at once (default
    `DEFAULT_VLLM_MAX_CONCURRENCY`).

    Kwarg translation: `max_new_tokens` -> `max_tokens` (default
    `DEFAULT_MAX_NEW_TOKENS`), `min_new_tokens` -> `min_tokens`;
    `temperature` (0 is greedy), `top_p`, `seed` and `stop` map one to one
    onto `SamplingParams`. Other kwargs pass through as `SamplingParams`
    fields.

    Aborts: the adapter cancels a request through vLLM's own abort API, which
    takes effect at the next scheduler step, so a few extra tokens may be
    generated after the activation that triggered it. The adapter doesn't
    report which point stopped a request, so `GenerationOutput.aborted` is
    derived from the inline points' `signal_history`.

    Topology: the adapter was validated with a single worker. Tensor- or
    pipeline-parallel engines are refused unless you pass
    `allow_unsupported_executor=True` (see the vLLM adapter docs).
    """

    name = "vllm"

    def __init__(
        self, adapter: EngineAdapter | None = None, *, max_concurrency: int = DEFAULT_VLLM_MAX_CONCURRENCY
    ) -> None:
        self._adapter = adapter
        self._owns_adapter = adapter is None
        self._config_layers: int | None = None
        self.set_max_concurrency(max_concurrency)

    def create_adapter(self, model: Any, *, device: str | None, **model_kwargs: Any) -> EngineAdapter:
        if device is not None and not str(device).startswith("cuda"):
            raise ProbedModelConfigError(
                f"device={device!r} isn't supported by the 'vllm' backend: vLLM places the model on its own "
                "GPU(s). Drop device=, or use backend='hf' for CPU."
            )
        adapter = self._adapter if self._adapter is not None else _new_vllm_adapter()
        if model is not None:
            if not isinstance(model, (str, os.PathLike)):
                raise ProbingTypeError(
                    "backend='vllm' loads models by Hugging Face hub id or local path, got a "
                    f"{type(model).__name__}; pass a string, or use backend='hf' for a model object"
                )
            adapter.load_model(os.fspath(model), **model_kwargs)
        return adapter

    def supported_tensor_types(self) -> frozenset[TensorType]:
        from .adapters.vllm.support import supported_tensor_types

        return frozenset(t for t, reason in supported_tensor_types().items() if reason is None)

    def config_num_layers(self, model: Any, model_kwargs: Mapping[str, Any]) -> int | None:
        if not isinstance(model, (str, os.PathLike)):
            return None
        self._config_layers = _config_num_layers(
            os.fspath(model),
            revision=model_kwargs.get("revision"),
            trust_remote_code=bool(model_kwargs.get("trust_remote_code", False)),
        )
        return self._config_layers

    def num_layers(self, adapter: EngineAdapter) -> int | None:
        return self._config_layers

    def translate_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(kwargs)
        if "max_new_tokens" in out:
            out["max_tokens"] = out.pop("max_new_tokens")
        out.setdefault("max_tokens", DEFAULT_MAX_NEW_TOKENS)
        if "min_new_tokens" in out:
            out["min_tokens"] = out.pop("min_new_tokens")
        stop = out.get("stop")
        if stop is not None:
            out["stop"] = [stop] if isinstance(stop, str) else list(stop)
        return out

    def close(self, adapter: EngineAdapter) -> None:
        if not self._owns_adapter:
            return  # the caller owns the adapter
        shutdown = getattr(adapter, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _config_num_layers(model: str, *, revision: str | None, trust_remote_code: bool) -> int | None:
    """`num_hidden_layers` from the model's transformers config, read without
    loading weights. None (and no layer check) if the config can't be read."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001 -- the layer check is best-effort; vLLM reports load errors itself
        logger.debug("could not read the config of %r to check layer indices: %s", model, exc)
        return None
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        config = get_text_config()
    layers = getattr(config, "num_hidden_layers", None)
    return layers if isinstance(layers, int) else None


#: Backend name -> factory.
_BACKENDS: dict[str, Callable[[], _Backend]] = {"hf": HFBackend, "vllm": VLLMBackend}


def _resolve_backend(backend: str | EngineAdapter | _Backend) -> _Backend:
    if isinstance(backend, _Backend):
        return backend
    if isinstance(backend, EngineAdapter):
        from .adapters.hf import HFEngineAdapter

        if isinstance(backend, HFEngineAdapter):
            return HFBackend(backend)
        from .adapters.vllm import VLLMEngineAdapter  # imports no vllm

        if isinstance(backend, VLLMEngineAdapter):
            return VLLMBackend(backend)
        return _AdapterInstanceBackend(backend)
    if isinstance(backend, str):
        factory = _BACKENDS.get(backend)
        if factory is None:
            raise ProbingValueError(
                f"unknown backend {backend!r}.{did_you_mean(backend, _BACKENDS)} Available: "
                f"{', '.join(repr(b) for b in sorted(_BACKENDS))}, or pass an EngineAdapter instance."
            )
        return factory()
    raise ProbingTypeError(
        f"backend must be a backend name ({', '.join(repr(b) for b in sorted(_BACKENDS))}) or an EngineAdapter "
        f"instance, got {type(backend).__name__}"
    )


# ---------------------------------------------------------------------------
# Spec handling and validation
# ---------------------------------------------------------------------------


def _load_spec(spec: SpecInput) -> ProbeSpec:
    if spec is None:
        return ProbeSpec(version="1", extraction_points=())
    if isinstance(spec, (ProbeSpec, os.PathLike, str, Mapping)):
        return load_spec(spec)
    points = list(spec)
    for point in points:
        if not isinstance(point, ExtractionPoint):
            raise ProbingTypeError(
                "spec must be a YAML path, a YAML string, a dict, a ProbeSpec or a list of "
                f"ExtractionPoint; the list contains a {type(point).__name__} ({point!r:.60})"
            )
    names = [p.name for p in points]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ProbedModelConfigError(
            f"duplicate extraction point name(s): {', '.join(map(repr, duplicates))}. "
            "Names must be unique; rename one of each pair."
        )
    return ProbeSpec(version="1", extraction_points=tuple(points))


def _validate_probes(spec: ProbeSpec, router: Router) -> None:
    """Every `probe_type` resolves and implements the declared `probe_kind`."""
    for point in spec:
        try:
            # The Router's own registration-time checks, run up front so a
            # mistake surfaces at construction, not on the first generate().
            router._validate_extraction_point(point)
            router._resolve_probe_factory(point)
        except RouterError as exc:
            message = str(exc)
            if isinstance(exc.__cause__, ProbeNotFoundError):
                message += f" Or pass probes={{{point.probe_type!r}: YourProbeClass}} to ProbedModel."
            raise ProbedModelConfigError(message) from exc


def _validate_tensor_types(spec: ProbeSpec, backend: _Backend) -> None:
    supported = backend.supported_tensor_types()
    if supported is None:
        return
    for point in spec:
        if point.tensor_type not in supported:
            names = ", ".join(sorted(t.value for t in supported))
            raise ProbedModelConfigError(
                f"extraction point {point.name!r}: tensor_type={point.tensor_type.value!r} is not supported "
                f"by the {backend.name!r} backend. Use one of: {names}. `undercurrent inspect-model MODEL` "
                "shows which tensor types each backend supports."
            )


def _validate_layers(spec: ProbeSpec, num_layers: int | None, model: Any = None) -> None:
    if num_layers is None:
        return
    model_name = os.fspath(model) if isinstance(model, (str, os.PathLike)) else None
    for point in spec:
        bad = [layer for layer in point.layers if not 0 <= layer < num_layers]
        if bad:
            of_model = f" for {model_name}" if model_name else ""
            raise ProbedModelConfigError(
                f"extraction point {point.name!r}: layer {bad[0]} is out of range{of_model} ({num_layers} "
                f"layers, valid 0..{num_layers - 1}). Use `undercurrent inspect-model "
                f"{model_name or 'MODEL'}` to list layers."
            )


# ---------------------------------------------------------------------------
# ProbedModel
# ---------------------------------------------------------------------------


class ProbedModel:
    """A model with probes attached. Build it with `ProbedModel.from_pretrained(...)`
    (the constructor takes the same arguments).

    ```python
    from undercurrent import ProbedModel

    with ProbedModel.from_pretrained("gpt2", spec="probes.yaml") as model:
        out = model.generate("The weather today is", max_new_tokens=20)
    print(out.text, out.probe_results)
    ```

    Use it as a context manager, or call `close()` when done.

    Args:
        model: a Hugging Face hub id or local path, or an already-built model
            object (then pass `tokenizer=`). May be None only when `backend`
            is an `EngineAdapter` that already has a model loaded.
        spec: the extraction points: a YAML file path, a YAML string, a dict,
            a `ProbeSpec`, a list of `ExtractionPoint`, or None for no probes.
        probes: `{probe_type: Probe subclass | ProbeFactory}`. Defaults to the
            global registry (`@register_probe`). Can't be combined with `router=`.
        backend: `"hf"` (transformers, the default), `"vllm"` (needs vLLM
            installed in the environment, in a supported version; vLLM places
            the model itself and `**model_kwargs` are vLLM engine args), or
            an `EngineAdapter` instance (used as-is; the caller keeps owning
            it).
        device: device for the model, e.g. `"cpu"` or `"cuda"` (HF default:
            cpu). vLLM places the model itself; leave it None.
        max_concurrency: how many prompts of a batch `generate()` runs at
            once. Defaults to the backend's own value (HF: 1, vLLM:
            `DEFAULT_VLLM_MAX_CONCURRENCY`); HF only supports 1.
        on_result: `callback(GenerationOutput)`, called once per finished
            generation on the thread that finished it. Exceptions it raises
            are logged on the `undercurrent` logger and never break generation.
        log_sink: advanced; a [`LogSink`][undercurrent.sinks.LogSink] (e.g. a
            `FileLogSink`) for async ("observe mode") extraction points.
        metrics_sink: advanced; a [`MetricsSink`][undercurrent.router.MetricsSink]
            for async-binding metrics.
        router_kwargs: advanced; forwarded to [`Router`][undercurrent.router.Router] (`worker_pool_size`,
            `default_queue_depth`, `default_overflow_policy`, `drain_timeout`,
            `default_intervention_policy`, `circuit_breaker_threshold`).
        router: advanced; use this `Router` as-is. The caller owns it, so
            `close()` doesn't shut it down. Can't be combined with `probes=`
            or `router_kwargs=`. `log_sink`/`metrics_sink` are attached to it.
        **model_kwargs: forwarded to the backend's model loading (for HF,
            `tokenizer=` plus `from_pretrained` kwargs such as `torch_dtype`;
            for vLLM, engine args such as `gpu_memory_utilization` and
            `max_model_len`).

    Raises:
        ProbedModelConfigError: a `probe_type` doesn't resolve, a `probe_kind`
            doesn't match its probe, a `tensor_type` isn't supported by the
            backend, or a layer index is out of range.
        MissingDependencyError: `backend="vllm"` but vLLM isn't installed.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        spec: SpecInput = None,
        probes: Mapping[str, type[Probe] | ProbeFactory] | ProbeRegistry | None = None,
        backend: str | EngineAdapter = "hf",
        device: str | None = None,
        on_result: OnResult | None = None,
        log_sink: SupportsLogSink | None = None,
        metrics_sink: MetricsSink | None = None,
        router_kwargs: Mapping[str, Any] | None = None,
        router: Router | None = None,
        max_concurrency: int | None = None,
        **model_kwargs: Any,
    ) -> None:
        self._closed = False
        self._on_result = on_result
        self._spec = _load_spec(spec)
        self._backend = _resolve_backend(backend)
        if max_concurrency is not None:
            self._backend.set_max_concurrency(max_concurrency)

        if router is not None:
            if probes is not None or router_kwargs is not None:
                raise ProbingValueError(
                    "router= can't be combined with probes= or router_kwargs=: configure the Router you pass "
                    "in directly (Router(probes, **router_kwargs)), or drop router= and let ProbedModel build one"
                )
            self._router = router
            self._owns_router = False
            if metrics_sink is not None:
                router.attach_metrics_sink(metrics_sink)
        else:
            kwargs = dict(router_kwargs or {})
            if metrics_sink is not None:
                if kwargs.get("metrics_sink") is not None:
                    raise ProbingValueError(
                        "metrics_sink is given both directly and in router_kwargs; pass it once "
                        "(ProbedModel(metrics_sink=...) is enough)"
                    )
                kwargs["metrics_sink"] = metrics_sink
            self._router = Router(probes, **kwargs)
            self._owns_router = True
        if log_sink is not None:
            self._router.attach_log_sink(log_sink)

        self._inflight: dict[str, dict[str, ProbeResult] | None] = {}
        self._inflight_lock = threading.Lock()
        self._remove_listener = self._router.on_request_end(self._capture_results)
        self._slots = threading.BoundedSemaphore(max(1, self._backend.max_concurrency))
        self._adapter: EngineAdapter | None = None

        try:
            # Spec-only checks first, so a typo fails before a model download.
            _validate_probes(self._spec, self._router)
            _validate_tensor_types(self._spec, self._backend)
            _validate_layers(self._spec, self._backend.config_num_layers(model, model_kwargs), model)
            self._adapter = self._backend.create_adapter(model, device=device, **model_kwargs)
            _validate_layers(self._spec, self._backend.num_layers(self._adapter), model)
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_pretrained(cls, model: Any = None, **kwargs: Any) -> ProbedModel:
        """Load `model` and attach probes; see the class docstring for the arguments."""
        return cls(model, **kwargs)

    # -- public accessors ----------------------------------------------------

    @property
    def router(self) -> Router:
        """The `Router` dispatching activations to probes (advanced use)."""
        return self._router

    @property
    def adapter(self) -> EngineAdapter:
        """The engine adapter running the model (advanced use)."""
        if self._adapter is None:
            raise ProbingRuntimeError(
                "ProbedModel has no adapter (construction failed or it was closed); create a new ProbedModel"
            )
        return self._adapter

    @property
    def spec(self) -> ProbeSpec:
        """The resolved extraction points every generation is probed with."""
        return self._spec

    @property
    def max_concurrency(self) -> int:
        """How many prompts of a batch `generate()` runs at once (1 for HF)."""
        return self._backend.max_concurrency

    # -- generation -------------------------------------------------------------

    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = ...,
        temperature: float | None = ...,
        top_p: float | None = ...,
        seed: int | None = ...,
        stop: str | Sequence[str] | None = ...,
        return_exceptions: bool = ...,
        **backend_kwargs: Any,
    ) -> GenerationOutput: ...

    @overload
    def generate(
        self,
        prompt: Sequence[str],
        *,
        max_new_tokens: int | None = ...,
        temperature: float | None = ...,
        top_p: float | None = ...,
        seed: int | None = ...,
        stop: str | Sequence[str] | None = ...,
        return_exceptions: Literal[False] = ...,
        **backend_kwargs: Any,
    ) -> list[GenerationOutput]: ...

    @overload
    def generate(
        self,
        prompt: Sequence[str],
        *,
        max_new_tokens: int | None = ...,
        temperature: float | None = ...,
        top_p: float | None = ...,
        seed: int | None = ...,
        stop: str | Sequence[str] | None = ...,
        return_exceptions: Literal[True],
        **backend_kwargs: Any,
    ) -> list[GenerationOutput | Exception]: ...

    def generate(
        self,
        prompt: str | Sequence[str],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        return_exceptions: bool = False,
        **backend_kwargs: Any,
    ) -> GenerationOutput | list[GenerationOutput] | list[GenerationOutput | Exception]:
        """Generate for one prompt (returns a `GenerationOutput`) or a list of
        prompts (returns a list, same order).

        A list runs up to `max_concurrency` prompts at once (vLLM batches
        them; HF runs them one by one). By default one failing prompt raises
        its exception for the whole call (prompts not started yet are
        cancelled). With `return_exceptions=True` the exception is put in that
        prompt's slot of the returned list instead. A single string prompt
        always raises.

        `temperature=0` is greedy decoding, `temperature > 0` samples. `seed`
        makes sampling reproducible. `stop` ends generation at a string (not
        included in the text). Other keyword arguments go to the backend
        unchanged (for HF, `model.generate(...)` kwargs such as
        `repetition_penalty`).
        """
        if self._closed:
            raise ProbingRuntimeError(
                "ProbedModel is closed; generate() only works before close() (or inside its `with` block). "
                "Create a new ProbedModel."
            )
        normalised = {
            key: value
            for key, value in (
                ("max_new_tokens", max_new_tokens),
                ("temperature", temperature),
                ("top_p", top_p),
                ("seed", seed),
                ("stop", stop),
            )
            if value is not None
        }
        kwargs = {**backend_kwargs, **normalised}

        if isinstance(prompt, str):
            return self._generate_one(prompt, kwargs)
        return self._generate_many(list(prompt), kwargs, return_exceptions)

    def _generate_many(
        self, prompts: list[str], kwargs: Mapping[str, Any], return_exceptions: bool
    ) -> list[GenerationOutput | Exception]:
        workers = min(len(prompts), self._backend.max_concurrency)
        if workers <= 1:
            outputs: list[GenerationOutput | Exception] = []
            for p in prompts:
                try:
                    outputs.append(self._generate_one(p, kwargs))
                except Exception as exc:
                    if not return_exceptions:
                        raise
                    outputs.append(exc)
            return outputs

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="undercurrent-generate") as pool:
            futures = [pool.submit(self._generate_one, p, kwargs) for p in prompts]
            if return_exceptions:
                wait(futures)
                outputs = []
                for f in futures:
                    error = f.exception()
                    if error is None:
                        outputs.append(f.result())
                    elif isinstance(error, Exception):
                        outputs.append(error)
                    else:
                        raise error
                return outputs
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            failed = [f for f in futures if f in done and f.exception() is not None]
            if failed:
                for f in futures:
                    f.cancel()  # only prompts that haven't started; running ones finish
                failed[0].result()  # raises
            return [f.result() for f in futures]

    def _generate_one(self, prompt: str, kwargs: Mapping[str, Any]) -> GenerationOutput:
        adapter = self.adapter
        request_id = uuid.uuid4().hex
        with self._slots:
            with self._inflight_lock:
                self._inflight[request_id] = None
            try:
                adapter.register_extraction(request_id, list(self._spec))
                try:
                    text = self._backend.run(adapter, request_id, prompt, kwargs, self._router)
                    abort = self._backend.abort_info(adapter, request_id)
                finally:
                    adapter.unregister_extraction(request_id)
            finally:
                with self._inflight_lock:
                    results = self._inflight.pop(request_id, None) or {}

        output = self._build_output(prompt, request_id, text, results, abort)
        if self._on_result is not None:
            try:
                self._on_result(output)
            except Exception:  # noqa: BLE001 -- a broken callback must not lose the output
                logger.exception(
                    "ProbedModel on_result callback %r raised for request_id=%r", self._on_result, request_id
                )
        return output

    def _capture_results(self, request_id: str, results: dict[str, ProbeResult]) -> None:
        # Router.on_request_end listener. A caller-owned router may serve other
        # requests too; only keep ours.
        with self._inflight_lock:
            if request_id in self._inflight:
                self._inflight[request_id] = results

    def _build_output(
        self,
        prompt: str,
        request_id: str,
        text: str,
        results: dict[str, ProbeResult],
        abort: tuple[str | None, ProbeSignal] | None,
    ) -> GenerationOutput:
        if abort is None:
            abort = self._abort_from_history(results)
        if abort is None:
            return GenerationOutput(text=text, prompt=prompt, request_id=request_id, probe_results=results)
        point, signal = abort
        return GenerationOutput(
            text=text,
            prompt=prompt,
            request_id=request_id,
            probe_results=results,
            aborted=True,
            abort_reason=_abort_reason(point, signal),
            abort_signal=signal,
            abort_point=point,
        )

    def _abort_from_history(self, results: Mapping[str, ProbeResult]) -> tuple[str, ProbeSignal] | None:
        """Fallback when the adapter doesn't report its stop signal: the
        earliest ABORT in any inline point's `signal_history`."""
        first: tuple[str, ProbeSignal] | None = None
        for point in self._spec:
            if point.execution_mode != ExecutionMode.INLINE or point.name not in results:
                continue
            for signal in results[point.name].signal_history:
                if signal.action == ProbeAction.ABORT and (first is None or signal.timestamp < first[1].timestamp):
                    first = (point.name, signal)
                    break
        return first

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Release the adapter and shut down the Router, unless the caller
        passed them in (`backend=<EngineAdapter>`, `router=`): those stay
        untouched apart from removing this model's results listener.
        Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._remove_listener()
        if self._adapter is not None:
            try:
                self._backend.close(self._adapter)
            finally:
                self._adapter = None
                if self._owns_router:
                    self._router.shutdown(wait=True)
        elif self._owns_router:
            self._router.shutdown(wait=True)

    def __enter__(self) -> ProbedModel:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{len(self._spec)} extraction point(s)"
        return f"ProbedModel(backend={self._backend.name!r}, {state})"


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_VLLM_MAX_CONCURRENCY",
    "GenerationOutput",
    "ProbedModel",
    "ProbedModelConfigError",
]

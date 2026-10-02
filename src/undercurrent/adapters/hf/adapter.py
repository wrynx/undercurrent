"""HFEngineAdapter: the `EngineAdapter` implementation for HuggingFace
`transformers` models, driven through a single sequential `model.generate()`
call per request.

Scope (read this before using the class)
-----------------------------------------
This is the platform's reference adapter for the *cleanest* intervention
semantics (see `stopping_criteria.py`'s module docstring): a single
sequential decode loop, one request at a time, no batching. It deliberately
does NOT support:
    - Concurrent `generate()` calls (unlike `undercurrent.adapters.vllm`, there is
      no event-loop/scheduler layer here to interleave them) -- calling
      `generate()` while another request is still active raises
      `HFAdapterLimitationError`.
    - Beam search / any decoding strategy that keeps more than one
      candidate sequence alive at once, or any batch size > 1. Position
      tracking below assumes exactly one sequence advancing one token at a
      time.
    - `tensor_type="kv"` extraction points, for the same reason
      `undercurrent.adapters.vllm` doesn't support them (see that package's
      `worker_extension.py`): KV cache isn't a per-layer forward-pass
      tensor with one row per token the way residual_stream/attn_out/
      mlp_out are.
These are documented, explicit limitations (raising clearly), not silent
correctness gaps -- consistent with this platform's existing adapters.

Interception strategy
----------------------
Mirrors `undercurrent.adapters.vllm.worker_extension`'s choice for the same
reason: `torch.nn.Module.register_forward_hook` on the model's decoder
layer / attention / MLP submodules, found via a short list of plausible
attribute paths (`transformer.h` for GPT-2-style models, `model.layers` for
Llama-family models, ...). A hook fires once per submodule per forward
call; this adapter also wraps `model.forward` itself (see
`_install_forward_wrapper`) purely to count decode steps -- see
`_ActiveGeneration`'s docstring for why a call-counter, not tensor shape, is
what actually distinguishes the prefill call from a decode-step call.

Intervention wiring
--------------------
Each hook calls `router.route(record)` synchronously (exactly like vLLM's
`_emit`) and, for an inline extraction point whose signal comes back
`action="abort"`, sets `_ActiveGeneration.should_stop`.
`ProbingStoppingCriteria` (installed on every `generate()` call) reads that
flag once per decode step -- see that module's docstring for why no
additional timeout handling is needed there: `router.route()` already
enforces `InterventionPolicy.timeout_ms` before returning, and this
adapter's single-sequential-request model means that wait is scoped to
exactly the one request it's for, unlike vLLM's continuously-batched case.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...core import ProbeAction, ProbeSignal, RequestContext
from ...router import Router
from ...spec import ActivationRecord, ExecutionMode, ExtractionPoint, TensorType
from .._optional import require_torch, require_transformers
from ..base import EngineAdapter
from .errors import HFAdapterLimitationError
from .introspect import find_attention, find_decoder_layers, find_mlp, supported_tensor_types
from .stopping_criteria import ProbingStoppingCriteria

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime torch dependency
    import torch

ISSUES = "https://github.com/wrynx/undercurrent/issues"

_UNSUPPORTED_TENSOR_TYPES = tuple(t for t, reason in supported_tensor_types().items() if reason is not None)


@dataclass
class _ActiveGeneration:
    """Per-`generate()`-call state shared between the forward hooks (see
    `HFEngineAdapter._make_forward_hook`), the forward wrapper (see
    `HFEngineAdapter._install_forward_wrapper`), and
    `ProbingStoppingCriteria`. Exactly one of these is ever live at a time
    -- see the module docstring's "no concurrent generate()" limitation.

    `step_count` (not tensor shape) is what distinguishes the prefill call
    from a decode-step call: with a 1-token prompt, the prefill call's
    hidden-states tensor also has seq_len==1, indistinguishable from a
    decode step by shape alone. Counting calls instead ("the first forward
    this generate() call makes is always the prefill, regardless of how
    long the prompt is") is unambiguous. `generated_count` is bumped by
    exactly one per decode-shaped call, from `_install_forward_wrapper`'s
    wrapper -- never from inside a per-layer hook, since multiple hooks
    (one per layer/submodule) fire per single decode step and would
    otherwise double- or triple-count it.
    """

    request_id: str
    router: Router
    extraction_points: tuple[ExtractionPoint, ...]
    prompt_len: int
    step_count: int = 0
    generated_count: int = 0
    current_step_is_decode: bool = False
    should_stop: bool = False
    stop_signal: ProbeSignal | None = None
    stop_point: str | None = None


class HFEngineAdapter(EngineAdapter):
    """The [`EngineAdapter`][undercurrent.adapters.EngineAdapter] for Hugging Face transformers models.

    Captures activations with forward hooks on the decoder layers and their
    attention and MLP submodules during ``model.generate()``, and stops
    generation through a ``StoppingCriteria`` when an inline probe aborts.
    An abort takes effect at the next decode step, and a
    ``block_until_signal`` wait only ever holds up the one request it is for.

    Known limitations, which raise
    [`HFAdapterLimitationError`][undercurrent.adapters.hf.HFAdapterLimitationError]:

    - one ``generate()`` at a time (no concurrent calls, batch size 1);
    - no beam search or other decoding that keeps several sequences alive;
    - no ``tensor_type="kv"``;
    - architectures whose decoder-layer list isn't found (GPT-2-style
      ``transformer.h`` and Llama-style ``model.layers`` are).

    ```python
    adapter = HFEngineAdapter()
    adapter.load_model("gpt2")
    adapter.register_extraction(request_id, extraction_points)
    text = adapter.generate(request_id, prompt, {"max_new_tokens": 32}, router)
    adapter.unregister_extraction(request_id)
    ```
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._model_name: str | None = None
        self._pending_extraction_points: dict[str, list[ExtractionPoint]] = {}
        self._bound_router: Router | None = None
        self._active: _ActiveGeneration | None = None
        self._hooks_installed = False
        self._forward_wrapped = False
        self._handles: list[Any] = []
        self._last_stop: tuple[str, str | None, ProbeSignal] | None = None

    @property
    def model(self) -> Any:
        """The loaded `transformers` model (None before `load_model`)."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """The loaded tokenizer (None before `load_model`)."""
        return self._tokenizer

    @property
    def num_layers(self) -> int:
        """Number of hookable decoder layers; valid `layers` indices are `0..num_layers-1`."""
        if self._model is None:
            raise HFAdapterLimitationError("num_layers is only known after load_model(); call load_model(model) first")
        return len(self._find_decoder_layers())

    def last_stop(self, request_id: str) -> tuple[str | None, ProbeSignal] | None:
        """`(extraction_point_name, signal)` of the inline abort that stopped
        the most recent `generate()` call, if that call was for `request_id`
        and was aborted; otherwise None."""
        if self._last_stop is None or self._last_stop[0] != request_id:
            return None
        return self._last_stop[1], self._last_stop[2]

    def close(self) -> None:
        """Remove every hook this adapter installed on the model and restore
        its original `forward`. The adapter can't generate afterwards."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._forward_wrapped and self._model is not None:
            # The wrapper was set as an instance attribute; deleting it
            # re-exposes the class's own bound `forward`.
            if "forward" in vars(self._model):
                del self._model.forward
            self._forward_wrapped = False
        self._hooks_installed = False
        self._model = None

    # -----------------------------------------------------------------
    # EngineAdapter contract
    # -----------------------------------------------------------------

    def load_model(self, model_name_or_path: Any, **kwargs: Any) -> None:
        """Load a model and its tokenizer.

        Args:
            model_name_or_path: a Hugging Face hub id or local path (loaded with
                ``AutoModelForCausalLM`` / ``AutoTokenizer``), or an
                already-built model object, which then needs ``tokenizer=``.
            **kwargs: ``device=`` (default ``"cpu"``) and ``tokenizer=``; the rest
                is passed to ``from_pretrained`` (e.g. ``torch_dtype``).
        """
        require_torch()
        transformers_mod = require_transformers()

        device = kwargs.pop("device", "cpu")
        tokenizer = kwargs.pop("tokenizer", None)

        if isinstance(model_name_or_path, str):
            self._model = transformers_mod.AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            self._tokenizer = tokenizer or transformers_mod.AutoTokenizer.from_pretrained(model_name_or_path)
            self._model_name = model_name_or_path
        else:
            if tokenizer is None:
                raise HFAdapterLimitationError(
                    f"load_model() was given an already-constructed {type(model_name_or_path).__name__} (not a "
                    "string model_name_or_path) but no tokenizer= kwarg -- a tokenizer "
                    "can't be inferred from a bare model instance. Pass tokenizer=AutoTokenizer.from_pretrained(...), "
                    "or pass the model's hub id or local path instead."
                )
            self._model = model_name_or_path
            self._tokenizer = tokenizer
            self._model_name = type(model_name_or_path).__name__

        self._model.to(device)
        self._model.eval()
        self._device = device
        self._ensure_hooks_installed()

    def register_extraction(self, request_id: str, extraction_points: list[ExtractionPoint]) -> None:
        if request_id in self._pending_extraction_points:
            raise HFAdapterLimitationError(
                f"register_extraction(): request_id {request_id!r} is already pending registration "
                "(generate() hasn't been called for it yet). Call generate() for it, or use a fresh request_id."
            )
        for ep in extraction_points:
            if ep.tensor_type in _UNSUPPORTED_TENSOR_TYPES:
                support = supported_tensor_types()
                usable = ", ".join(t.value for t, reason in support.items() if reason is None)
                raise HFAdapterLimitationError(
                    f"register_extraction(): extraction point {ep.name!r}: "
                    f"tensor_type={TensorType(ep.tensor_type).value!r} is not supported by the HF adapter "
                    f"({support[TensorType(ep.tensor_type)]}). Use one of: {usable}."
                )
        self._pending_extraction_points[request_id] = list(extraction_points)

    def generate(self, request_id: str, prompt: str, generation_kwargs: dict[str, Any], router: Router) -> str:
        if self._model is None:
            raise HFAdapterLimitationError("generate() called before load_model(); call load_model(model) first")
        extraction_points = self._pending_extraction_points.pop(request_id, None)
        if extraction_points is None:
            raise HFAdapterLimitationError(
                f"generate({request_id!r}): no prior register_extraction() call for this request_id. "
                "Call register_extraction(request_id, extraction_points) first (ProbedModel does this for you)."
            )
        if self._active is not None:
            raise HFAdapterLimitationError(
                "generate() called while another request is still active -- this reference "
                "adapter supports exactly one in-flight generate() call at a time (no "
                "batching/event-loop layer to interleave them, unlike undercurrent.adapters.vllm). "
                "Call generate() sequentially, one request at a time."
            )

        self._ensure_router_bound(router)

        torch = require_torch()
        transformers_mod = require_transformers()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        prompt_len = int(inputs["input_ids"].shape[1])

        request_ctx = RequestContext(
            request_id=request_id,
            prompt_metadata={"model": self._model_name, "prompt": prompt, "prompt_len": prompt_len},
            extraction_point_config=None,
        )
        router.register_request(request_id, extraction_points, request_ctx)

        state = _ActiveGeneration(
            request_id=request_id,
            router=router,
            extraction_points=tuple(extraction_points),
            prompt_len=prompt_len,
        )
        self._active = state
        self._last_stop = None
        try:
            stopping_criteria = transformers_mod.StoppingCriteriaList([ProbingStoppingCriteria(state)])
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    stopping_criteria=stopping_criteria,
                    **generation_kwargs,
                )
            generated_ids = output_ids[0][prompt_len:]
            return self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        finally:
            if state.should_stop and state.stop_signal is not None:
                self._last_stop = (request_id, state.stop_point, state.stop_signal)
            self._active = None
            # Finalizes every probe for this request (on_end) and tears down router-side
            # state. Safe even if generation was aborted mid-stream -- see Router.end_request's
            # own docstring on why an aborted request's trajectory probe still gets a clean finish.
            router.end_request(request_id)

    def unregister_extraction(self, request_id: str) -> None:
        self._pending_extraction_points.pop(request_id, None)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _ensure_router_bound(self, router: Router) -> None:
        if self._bound_router is router:
            return
        if self._bound_router is not None:
            raise HFAdapterLimitationError(
                "a different Router instance was already bound to this adapter -- "
                "HFEngineAdapter supports exactly one Router for its whole lifetime. Pass the same Router to "
                "every generate() call, or create a new HFEngineAdapter for the other Router."
            )
        self._bound_router = router

    def _ensure_hooks_installed(self) -> None:
        if self._hooks_installed:
            return
        decoder_layers = self._find_decoder_layers()
        for layer_idx, layer_module in decoder_layers.items():
            self._handles.append(
                layer_module.register_forward_hook(self._make_forward_hook(layer_idx, TensorType.RESIDUAL_STREAM))
            )
            attn = find_attention(layer_module)
            if attn is not None:
                self._handles.append(
                    attn.register_forward_hook(self._make_forward_hook(layer_idx, TensorType.ATTN_OUT))
                )
            mlp = find_mlp(layer_module)
            if mlp is not None:
                self._handles.append(mlp.register_forward_hook(self._make_forward_hook(layer_idx, TensorType.MLP_OUT)))
        self._install_forward_wrapper()
        self._hooks_installed = True

    def _find_decoder_layers(self) -> dict[int, torch.nn.Module]:
        """Locate the loaded model's decoder layers; see
        `undercurrent.adapters.hf.introspect.find_decoder_layers`."""
        return find_decoder_layers(self._model)

    def _install_forward_wrapper(self) -> None:
        """Monkeypatch `self._model.forward` purely to count decode steps
        (see `_ActiveGeneration`'s docstring) -- mirrors
        `undercurrent.adapters.vllm.worker_extension`'s `execute_model` wrapper,
        same technique, same reason (no engine-supported "before/after one
        step" callback to hang this on). Applied once; idempotent."""
        if self._forward_wrapped:
            return
        original_forward = self._model.forward

        # functools.wraps keeps the model's forward signature visible through
        # `__wrapped__`: transformers' generate() validates model_kwargs
        # against `inspect.signature(model.forward)`, and older releases
        # (4.40) reject `attention_mask` if the signature is bare *args/**kwargs.
        @functools.wraps(original_forward)
        def _wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            active = self._active
            is_decode_step = active is not None and active.step_count > 0
            if active is not None:
                active.current_step_is_decode = is_decode_step
                active.step_count += 1
            result = original_forward(*args, **kwargs)
            if active is not None and is_decode_step:
                active.generated_count += 1
            return result

        self._model.forward = _wrapped_forward
        self._forward_wrapped = True

    def _make_forward_hook(self, layer_idx: int, tensor_type: TensorType):
        def _hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            active = self._active
            if active is None:
                return  # fail open: no request currently generating
            if not any(layer_idx in ep.layers and ep.tensor_type == tensor_type for ep in active.extraction_points):
                return  # cheap short-circuit before touching the (possibly large) output tensor

            tensor = _extract_captured_tensor(output, tensor_type, layer_idx)
            is_generated = active.current_step_is_decode
            seq_len = tensor.shape[1]

            for offset in range(seq_len):
                if is_generated:
                    token_index = active.prompt_len + active.generated_count
                    generated_index = active.generated_count
                else:
                    token_index = offset
                    generated_index = None

                for ep in active.extraction_points:
                    if layer_idx not in ep.layers or ep.tensor_type != tensor_type:
                        continue
                    if not ep.matches(
                        token_index,
                        is_generated,
                        active.prompt_len,
                        generated_index=generated_index,
                        num_generated_total=None,
                    ):
                        continue
                    self._emit(active, offset, ep, layer_idx, tensor_type, tensor, token_index, is_generated)

        return _hook

    def _emit(
        self,
        active: _ActiveGeneration,
        offset: int,
        ep: ExtractionPoint,
        layer_idx: int,
        tensor_type: TensorType,
        tensor: torch.Tensor,
        token_index: int,
        is_generated: bool,
    ) -> None:
        record = ActivationRecord(
            request_id=active.request_id,
            extraction_point_name=ep.name,
            layer=layer_idx,
            token_pos=token_index,
            tensor_type=tensor_type.value if hasattr(tensor_type, "value") else tensor_type,
            tensor=tensor[0, offset].detach().to("cpu"),
            is_generated=is_generated,
        )
        signal = active.router.route(record)
        if signal is not None and signal.action == ProbeAction.ABORT and ep.execution_mode == ExecutionMode.INLINE:
            if not active.should_stop:  # the first abort is the one that stopped generation
                active.stop_signal = signal
                active.stop_point = ep.name
            active.should_stop = True


def _extract_captured_tensor(output: Any, tensor_type: TensorType, layer_idx: int) -> torch.Tensor:
    """Pull the actual `[batch, seq, hidden]` tensor out of a layer/
    submodule's forward-hook `output`. See
    `undercurrent.adapters.vllm.worker_extension._extract_captured_tensor`'s
    docstring for why this refuses to guess past "first Tensor found" --
    same reasoning, independently duplicated for the same reason as
    `introspect.find_decoder_layers`."""
    torch = require_torch()

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise HFAdapterLimitationError(
        f"forward hook for layer={layer_idx} tensor_type={tensor_type!r} got an output of type "
        f"{type(output).__name__} it doesn't know how to extract a tensor from. This model's "
        "decoder layer / attention / mlp forward signature wasn't anticipated -- extend "
        "_extract_captured_tensor() rather than letting this silently mis-capture. Please report it "
        f"with the model id at {ISSUES}."
    )

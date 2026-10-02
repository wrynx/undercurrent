"""SeqIdMapper: translates one step of vLLM's continuously-batched execution
into `(request_id, absolute token_pos)` pairs -- the core translation problem
this adapter exists to solve.

Why this is hard (and doesn't exist in the HF adapter)
--------------------------------------------------------
In HF's plain sequential `generate()` loop, "the Nth generated token of this
request" is just a Python loop counter: one request, one sequence, one
tensor of shape `[1, seq_len, hidden]` per step. There is nothing to map.

vLLM's continuous batching throws that away. Every decoder-layer forward
pass operates on ONE flattened tensor of shape `[total_tokens_in_step,
hidden]`, concatenating whatever mix of prefill chunks and single decode
tokens the scheduler picked for THIS step across potentially many different
requests. Which row-range of that tensor belongs to which request_id --
and at what absolute position within that request's own sequence -- changes
every step, because:
  - requests join the batch (a new prompt starts prefilling) and leave it
    (finish, or get preempted) independently of one another;
  - a finished request's row slot in vLLM's persistent batch gets reused by
    a newly-admitted request, so "row index 3" means a different request_id
    on step N than it did on step N-1;
  - prefill can be chunked (a long prompt spread across several steps, each
    contributing a different-sized row range) before decode (one row per
    step) even begins.

So "row -> (request_id, token_pos)" cannot be a static assignment computed
once; it must be recomputed from that step's scheduler metadata, every step.
That recomputation is exactly what `SeqIdMapper.resolve_step()` does.

This module is intentionally pure Python with no vLLM or torch import: the
inputs it needs (`StepBatchMetadata`) are plain data, so this translation
logic is unit-testable against hand-built/mocked scheduler metadata without
a GPU, without vLLM installed, and without running any real model. Reading
that metadata OFF a live vLLM `ModelRunner` is a separate, vLLM-version-
coupled concern -- see `introspection.py`.

Internal and experimental: its data model mirrors undocumented vLLM
scheduler internals, so it is not part of the public API (see
docs/api-stability.md) and changes whenever vLLM does.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ...errors import ProbingError


class SeqMapperError(ProbingError):
    """Raised for invalid SeqIdMapper usage or internally inconsistent step metadata."""


@dataclass(frozen=True)
class StepBatchMetadata:
    """One step's row layout for vLLM's flattened multi-request batch tensor.

    All three tuples are parallel and describe the batch in the SAME row
    order the model actually flattens it in for this step (see
    `introspection.py` for how that order is read off live vLLM internals --
    this type itself doesn't care where the values came from).

    Attributes:
        req_ids: request ids in this step's row order. May repeat across
            different `StepBatchMetadata` instances (same request, later
            step) but never within one instance.
        num_scheduled_tokens: for each entry in `req_ids`, how many
            consecutive rows of this step's flattened tensor that request
            contributes (1 for an ordinary decode step; >1 for a
            prefill/chunked-prefill step). `sum(num_scheduled_tokens) ==`
            the row count of the step's captured activation tensor.
        num_computed_tokens_before_step: for each entry in `req_ids`, how
            many tokens of that request (prompt + previously generated) had
            already been computed BEFORE this step -- i.e. the absolute
            sequence index of that request's first row in this step.
    """

    req_ids: tuple[str, ...]
    num_scheduled_tokens: tuple[int, ...]
    num_computed_tokens_before_step: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.req_ids)
        if len(self.num_scheduled_tokens) != n or len(self.num_computed_tokens_before_step) != n:
            raise SeqMapperError(
                "StepBatchMetadata: req_ids, num_scheduled_tokens, and "
                "num_computed_tokens_before_step must all be the same length, got "
                f"{n}, {len(self.num_scheduled_tokens)}, {len(self.num_computed_tokens_before_step)}"
            )
        if len(set(self.req_ids)) != n:
            raise SeqMapperError(f"StepBatchMetadata: req_ids must be unique within one step, got {self.req_ids!r}")
        for req_id, n_tok in zip(self.req_ids, self.num_scheduled_tokens):
            if n_tok < 1:
                raise SeqMapperError(
                    f"StepBatchMetadata: num_scheduled_tokens for {req_id!r} must be >= 1, got {n_tok}"
                )
        for req_id, n_before in zip(self.req_ids, self.num_computed_tokens_before_step):
            if n_before < 0:
                raise SeqMapperError(
                    f"StepBatchMetadata: num_computed_tokens_before_step for {req_id!r} must be >= 0, got {n_before}"
                )

    @property
    def total_rows(self) -> int:
        return sum(self.num_scheduled_tokens)

    def row_range(self, request_id: str) -> tuple[int, int]:
        """Return `[start, end)` row bounds for `request_id` in this step's flattened tensor."""
        offset = 0
        for req_id, n_tok in zip(self.req_ids, self.num_scheduled_tokens):
            if req_id == request_id:
                return offset, offset + n_tok
            offset += n_tok
        raise SeqMapperError(f"StepBatchMetadata: no such request_id={request_id!r} in this step")


@dataclass(frozen=True)
class TokenRowMapping:
    """One row of a step's flattened batch tensor, resolved to a logical token position.

    Attributes:
        row: row index into the step's `[total_rows, hidden]` captured tensor.
        request_id: which request this row belongs to.
        token_index: absolute 0-based index in that request's full
            (prompt + generated) sequence -- directly usable as
            `ActivationRecord.token_pos`.
        is_generated: whether `token_index` falls in the generated portion.
        generated_index: 0-based index within the generated portion, or
            `None` for a prompt-token row. Required to evaluate
            `ExtractionPoint.matches()` for `generated[*]` / slice selectors.
    """

    row: int
    request_id: str
    token_index: int
    is_generated: bool
    generated_index: int | None


@dataclass
class _RequestState:
    prompt_len: int
    num_generated_so_far: int = 0


class SeqIdMapper:
    """Tracks per-request prompt length and translates each step's
    `StepBatchMetadata` into per-row logical token positions.

    Not thread-safe by itself -- callers (the worker extension) are expected
    to serialize access the same way vLLM serializes model-forward calls
    within one worker (single-threaded per step); see `worker_extension.py`.
    """

    def __init__(self) -> None:
        self._requests: dict[str, _RequestState] = {}

    def register_request(self, request_id: str, prompt_len: int) -> None:
        """Record `request_id`'s prompt length, before its first scheduled step.

        Must be called before `resolve_step()` ever sees a row for this
        request_id (i.e. before the request is submitted to the engine) --
        prefill happens on the very first step, so there is no "safe" later
        point to call this from.
        """
        if prompt_len < 1:
            raise SeqMapperError(f"register_request({request_id!r}): prompt_len must be >= 1, got {prompt_len}")
        if request_id in self._requests:
            raise SeqMapperError(f"register_request(): request_id {request_id!r} is already registered")
        self._requests[request_id] = _RequestState(prompt_len=prompt_len)

    def unregister_request(self, request_id: str) -> None:
        """Drop all tracked state for `request_id`. Safe to call once generation ends."""
        self._requests.pop(request_id, None)

    def is_registered(self, request_id: str) -> bool:
        return request_id in self._requests

    def num_generated_so_far(self, request_id: str) -> int:
        state = self._requests.get(request_id)
        if state is None:
            raise SeqMapperError(f"num_generated_so_far(): no such request_id={request_id!r}")
        return state.num_generated_so_far

    def prompt_len(self, request_id: str) -> int:
        state = self._requests.get(request_id)
        if state is None:
            raise SeqMapperError(f"prompt_len(): no such request_id={request_id!r}")
        return state.prompt_len

    def resolve_step(self, batch: StepBatchMetadata) -> Iterator[TokenRowMapping]:
        """Yield one `TokenRowMapping` per row of `batch`, in row order.

        Also advances each involved request's `num_generated_so_far`
        counter by however many of this step's rows were generated-portion
        tokens -- call this at most once per real step per request (calling
        it twice for the same step would double-count that advance).

        Raises `SeqMapperError` if `batch` references a request_id that was
        never `register_request()`-ed (or was already `unregister_request`-ed)
        -- this indicates a genuine bug in adapter wiring (a request reached
        the scheduler without its prompt length ever being recorded), not
        something to paper over with a guessed prompt length.
        """
        offset = 0
        for req_id, n_tok, n_before in zip(
            batch.req_ids, batch.num_scheduled_tokens, batch.num_computed_tokens_before_step
        ):
            state = self._requests.get(req_id)
            if state is None:
                raise SeqMapperError(
                    f"resolve_step(): request_id={req_id!r} has a scheduled row range but was never "
                    "registered via register_request() (or was already unregistered) -- this step's "
                    "metadata is inconsistent with adapter-side registration state"
                )

            step_generated_count = 0
            for local in range(n_tok):
                token_index = n_before + local
                is_generated = token_index >= state.prompt_len
                generated_index = (token_index - state.prompt_len) if is_generated else None
                if is_generated:
                    step_generated_count += 1
                yield TokenRowMapping(
                    row=offset + local,
                    request_id=req_id,
                    token_index=token_index,
                    is_generated=is_generated,
                    generated_index=generated_index,
                )
            state.num_generated_so_far += step_generated_count
            offset += n_tok

    def registered_request_ids(self) -> frozenset[str]:
        return frozenset(self._requests.keys())

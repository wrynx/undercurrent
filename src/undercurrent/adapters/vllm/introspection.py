"""Reads vLLM's live per-step batch layout off a running V1 `GPUModelRunner`
and turns it into the `StepBatchMetadata` that `seq_mapper.SeqIdMapper`
consumes.

*** THIS MODULE IS THE SEAM MOST LIKELY TO NEED RECONCILING AGAINST         ***
*** WHATEVER VLLM VERSION IS ACTUALLY INSTALLED. Everything it reads is an ***
*** internal, undocumented vLLM attribute, not a stable public API.        ***

What this assumes about vLLM's V1 architecture (as of the vLLM versions
current around this adapter's original authoring -- no vLLM install was
available in the environment this was written in to pin an exact version
against; see the vLLM adapter design doc's "Limitations" (docs/_legacy/)):

  - `model_runner.input_batch` is vLLM's persistent per-worker batch state
    (`vllm.v1.worker.gpu_input_batch.InputBatch` in the versions this was
    checked against). It exposes:
      - `req_ids: list[str | None]` -- row-slot -> request_id for every row
        slot currently resident in the persistent batch, in the SAME row
        order the model runner flattens tokens into the forward-pass input
        tensor. A `None` entry marks a reclaimed/empty slot (a finished
        request whose slot hasn't been reused yet).
      - `num_computed_tokens_cpu` -- an array, indexed the same way, giving
        each resident request's number of already-computed tokens as of
        BEFORE this step.
  - The scheduler output passed into `model_runner.execute_model(...)` each
    step (`vllm.v1.core.sched.output.SchedulerOutput` in those versions)
    exposes `num_scheduled_tokens: dict[str, int]`, mapping request_id to
    how many of that request's tokens are scheduled in THIS step (1 for a
    plain decode step, >1 for prefill / chunked prefill). A request resident
    in `input_batch` but not a key of this dict was not scheduled this step
    (e.g. it's finishing up async output processing) and contributes no rows.

  This module cross-references those two sources by request_id and assumes
  `input_batch.req_ids` order IS the flattening order the model actually
  sees. That assumption is inferred from reading vLLM's V1 gpu model runner
  source, not from a documented public contract -- if hook-captured
  `token_pos` values ever look wrong for a given vLLM version, this is the
  first place to check (e.g. by dumping `StepBatchMetadata` for a few steps
  and comparing against `SamplingParams`-forced deterministic output such as
  greedy decoding of a known prompt).

Explicitly out of scope / not read here:
  - Distributed (Ray / multiproc) executors, where `model_runner` lives in a
    worker process and this function must be invoked THERE (via the worker
    extension, which does run in-process to the model runner regardless of
    executor topology) -- reading input_batch itself is topology-agnostic;
    it's *delivering the resulting ActivationRecord back to the driver's
    Router* that isn't solved for that topology. See worker_extension.py.
  - `kv` tensor_type: KV cache lives in paged blocks addressed by slot
    mapping, not as a per-step forward-pass tensor with the same
    row-per-token shape as hidden_states/attn_out/mlp_out. Capturing it
    would need block-table-aware logic this module doesn't implement;
    registering a `kv` extraction point is rejected explicitly elsewhere
    (see `worker_extension.py`) rather than silently mis-mapped here.

Internal and experimental: this reads undocumented vLLM internals and is
not part of the public API (see docs/api-stability.md); it changes whenever
vLLM does.
"""

from __future__ import annotations

from typing import Any

from ...errors import ProbingError
from .._optional import COMPATIBILITY_DOC
from .seq_mapper import StepBatchMetadata

_VERSION_HINT = f"This usually means the installed vLLM is outside the supported range; see {COMPATIBILITY_DOC}."


class VLLMIntrospectionError(ProbingError):
    """Raised when the installed vLLM's internals don't match the shape this
    module was written against. Deliberately loud: guessing at a
    reconstructed StepBatchMetadata from a partially-missing/renamed
    attribute would silently produce wrong `token_pos` values downstream,
    which is worse than failing extraction outright."""


_REQUIRED_INPUT_BATCH_ATTRS = ("req_ids", "num_computed_tokens_cpu")
_REQUIRED_SCHEDULER_OUTPUT_ATTRS = ("num_scheduled_tokens",)


def check_introspection_compatible(model_runner: Any, scheduler_output: Any) -> None:
    """Fail loudly and early if the installed vLLM's internals don't match
    what `extract_step_batch_metadata` expects, instead of letting a missing
    attribute surface confusingly deep inside a forward hook.

    Call once at worker-extension setup time (e.g. on the first forward
    hook invocation) so a version mismatch is reported clearly rather than
    discovered as silently-wrong activation data.
    """
    input_batch = getattr(model_runner, "input_batch", None)
    if input_batch is None:
        raise VLLMIntrospectionError(
            "model_runner.input_batch is missing -- this adapter was written against vLLM's "
            "V1 GPUModelRunner (vllm.v1.worker.gpu_model_runner). If the installed vLLM uses a "
            "different internal structure (a V0 engine, or a V1 layout that has since been "
            "refactored), introspection.py needs reconciling against it before extraction "
            "results can be trusted. " + _VERSION_HINT
        )
    missing = [attr for attr in _REQUIRED_INPUT_BATCH_ATTRS if not hasattr(input_batch, attr)]
    if missing:
        raise VLLMIntrospectionError(
            f"model_runner.input_batch is missing expected attribute(s) {missing} -- the "
            "installed vLLM's InputBatch shape has likely changed since this adapter was "
            "written against it. Reconcile introspection.py's assumptions (see this module's "
            "docstring) against the installed vllm version before trusting any captured "
            "token_pos values. " + _VERSION_HINT
        )
    missing = [attr for attr in _REQUIRED_SCHEDULER_OUTPUT_ATTRS if not hasattr(scheduler_output, attr)]
    if missing:
        raise VLLMIntrospectionError(
            f"scheduler_output is missing expected attribute(s) {missing} -- the installed "
            "vLLM's SchedulerOutput shape has likely changed since this adapter was written "
            "against it. Reconcile introspection.py before trusting any captured token_pos values. " + _VERSION_HINT
        )


def extract_step_batch_metadata(model_runner: Any, scheduler_output: Any) -> StepBatchMetadata:
    """Build this step's `StepBatchMetadata` from live vLLM V1 internals.

    Must be called once per `model_runner.execute_model(scheduler_output)`
    invocation, with the SAME `scheduler_output` that call was given (see
    `worker_extension.py`'s `execute_model` wrapper, which is what actually
    calls this). Iterates `input_batch.req_ids` in row-slot order and keeps
    only slots that are (a) occupied (`req_id is not None`) and (b) actually
    scheduled this step (present in `scheduler_output.num_scheduled_tokens`)
    -- a resident-but-unscheduled row contributes no rows to this step's
    flattened forward-pass tensor.
    """
    input_batch = model_runner.input_batch
    num_scheduled_tokens: dict[str, int] = scheduler_output.num_scheduled_tokens
    num_computed_tokens_cpu = input_batch.num_computed_tokens_cpu

    req_ids = []
    num_scheduled = []
    num_computed_before = []
    for slot, req_id in enumerate(input_batch.req_ids):
        if req_id is None:
            continue
        n = num_scheduled_tokens.get(req_id)
        if n is None:
            continue
        req_ids.append(req_id)
        num_scheduled.append(int(n))
        num_computed_before.append(int(num_computed_tokens_cpu[slot]))

    return StepBatchMetadata(
        req_ids=tuple(req_ids),
        num_scheduled_tokens=tuple(num_scheduled),
        num_computed_tokens_before_step=tuple(num_computed_before),
    )

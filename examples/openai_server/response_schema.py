"""OpenAI-compatible response formatting for probed generation requests.

This module is a **pure, server-agnostic formatting layer**: plain Python
data in, plain Python (JSON-serializable) data out. It has no dependency on
vLLM, HTTP, or any web framework -- it is meant to be imported by whatever
serving layer sits in front of an inference engine (e.g. an
OpenAI-compatible vLLM wrapper) to turn a completed generation plus its
probe verdicts into response bodies / SSE chunks, without that server
needing to know anything about probing internals.

This is a **reference example**, not part of the installed ``undercurrent``
package: v0.1 ships no HTTP server (a built-in OpenAI-compatible server is on
the roadmap). Nothing here is vLLM-specific -- it only consumes the
engine-agnostic ``ExecutionMode``/``ProbeKind`` vocabulary from
``undercurrent.spec`` and plain per-probe verdict data, so any serving layer in
front of any engine adapter can copy or import it.

Typical usage from a serving layer::

    from response_schema import (
        GenerationOutcome, ProbeVerdict, ExecutionMode, ProbeKind,
        aggregate_trajectory_scores, build_completion_response,
        build_probing_result_sse_event, build_flagged_sse_chunk,
    )

    outcome = GenerationOutcome(
        completed=False, text=None, tokens_generated=17, max_tokens=256,
        aborted_reason="content_safety_flag",
    )
    verdicts = [
        ProbeVerdict(
            extraction_point_name="toxicity_probe",
            execution_mode=ExecutionMode.INLINE,
            probe_kind=ProbeKind.SINGLE_SHOT,
            flagged=True,
            score=0.94,
            evaluated=True,
        ),
    ]
    response = build_completion_response(outcome, verdicts)
    # -> {"flagged": True, "finish_reason": "content_flagged", ...}
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from undercurrent import ExecutionMode, ProbeKind

#: finish_reason used in place of the normal "stop"/"length" when an inline
#: probe aborted generation early.
FLAGGED_FINISH_REASON = "content_flagged"

#: SSE event name carrying the post-completion probing summary on the
#: streaming clean path.
PROBING_RESULT_EVENT = "probing_result"

__all__ = [
    "FLAGGED_FINISH_REASON",
    "PROBING_RESULT_EVENT",
    "GenerationOutcome",
    "ProbeVerdict",
    "TrajectoryAggregate",
    "aggregate_trajectory_scores",
    "build_completion_response",
    "build_probing_object",
    "build_flagged_sse_chunk",
    "build_probing_result_sse_event",
]


@dataclass(frozen=True)
class GenerationOutcome:
    """The outcome of one generation request, independent of probing.

    Attributes:
        completed: ``True`` if generation ran to a normal stop condition
            (EOS, ``max_tokens``, a stop sequence). ``False`` if it was
            aborted early because an inline probe flagged it.
        text: the generated completion text. ``None`` or ``""`` when
            suppressed (always the case when ``completed`` is ``False`` --
            flagged completions are never returned to the caller).
        tokens_generated: number of tokens actually produced before
            stopping (for whatever reason).
        max_tokens: the ``max_tokens`` that was requested for this
            generation. Used together with ``tokens_generated`` to report
            the tokens-saved metric on the flagged path.
        aborted_reason: free-form, caller-defined reason string for why
            generation was aborted (e.g. the extraction point name or a
            short code). Ignored when ``completed`` is ``True``. Not
            surfaced directly in the response body -- the flagging probe's
            own name/score (from the matching ``ProbeVerdict``) is what's
            reported -- but kept here for logging/debugging by the caller.
    """

    completed: bool
    text: str | None
    tokens_generated: int
    max_tokens: int
    aborted_reason: str | None = None

    @property
    def tokens_saved(self) -> int:
        """How many tokens generation was cut short by, vs. ``max_tokens``.

        Always ``>= 0`` under normal use (``tokens_generated`` should never
        exceed ``max_tokens``); clamped to 0 defensively if it somehow does.
        """
        return max(0, self.max_tokens - self.tokens_generated)


@dataclass(frozen=True)
class TrajectoryAggregate:
    """Aggregate summary of a trajectory probe's per-step scores.

    Produced by :func:`aggregate_trajectory_scores` from a full sequence of
    per-token scores. The full per-token trace itself is *not* this
    module's concern -- that goes to ``undercurrent.sinks`` -- only this
    two-number summary is surfaced in response bodies.
    """

    max_score: float
    final_score: float

    def as_dict(self) -> dict[str, float]:
        return {"max_score": self.max_score, "final_score": self.final_score}


def aggregate_trajectory_scores(scores: Sequence[float]) -> TrajectoryAggregate:
    """Reduce a trajectory probe's per-step scores to ``{max_score, final_score}``.

    Args:
        scores: per-step scores in emission order, e.g. one per generated
            token a ``trajectory`` probe has seen so far. Must be
            non-empty.

    Returns:
        A :class:`TrajectoryAggregate` with the maximum score seen and the
        most recent (last) score.

    Raises:
        ValueError: if ``scores`` is empty -- there is no meaningful
            aggregate for a trajectory probe that never ran. Use
            ``evaluated=False`` on the corresponding :class:`ProbeVerdict`
            instead of calling this with no scores.
    """
    if not scores:
        raise ValueError("aggregate_trajectory_scores requires at least one score")
    return TrajectoryAggregate(max_score=max(scores), final_score=scores[-1])


#: The score carried by a ProbeVerdict: either a single scalar (single_shot
#: probes, or a trajectory probe already reduced by the caller), or a
#: pre-aggregated TrajectoryAggregate. May be None for probes not evaluated,
#: or a probe kind that has no natural scalar score.
ProbeScore = float | TrajectoryAggregate | None


@dataclass(frozen=True)
class ProbeVerdict:
    """The outcome of one configured extraction point's probe for one request.

    One of these exists per extraction point in the request's ``ProbeSpec``
    (see ``undercurrent.spec``), regardless of whether it ever actually fired --
    an extraction point whose position was never reached (e.g. a
    ``generated[*]`` trajectory probe when generation ends before any
    generated token exists) is still represented, with ``evaluated=False``.

    Attributes:
        extraction_point_name: matches ``ExtractionPoint.name`` from
            ``undercurrent.spec``.
        execution_mode: ``inline`` (gating) or ``async`` (observational
            only, never gates) -- see ``undercurrent.spec.ExecutionMode``.
        probe_kind: ``single_shot`` or ``trajectory`` -- see
            ``undercurrent.spec.ProbeKind``.
        flagged: whether this probe's verdict should be treated as a
            positive detection. For ``execution_mode=async`` probes this is
            informational only (async probes never gate generation).
        score: a scalar score (``float``), a :class:`TrajectoryAggregate`
            (for a ``trajectory`` probe reporting ``max_score``/
            ``final_score``; typically produced via
            :func:`aggregate_trajectory_scores`), or ``None`` if no score is
            available.
        evaluated: ``False`` if this extraction point's position was never
            reached during generation (e.g. request ended before a
            ``generated[*]`` trajectory probe could run) -- ``flagged`` and
            ``score`` should be treated as meaningless (their default
            falsy/``None`` values) when this is ``False``.
    """

    extraction_point_name: str
    execution_mode: ExecutionMode
    probe_kind: ProbeKind
    flagged: bool = False
    score: ProbeScore = None
    evaluated: bool = True

    def score_field(self) -> Any:
        """The JSON-serializable form of ``score`` for response bodies.

        A plain float/``None`` passes through unchanged; a
        :class:`TrajectoryAggregate` becomes its ``{max_score,
        final_score}`` dict.
        """
        if isinstance(self.score, TrajectoryAggregate):
            return self.score.as_dict()
        return self.score


def _find_flagging_verdict(verdicts: Sequence[ProbeVerdict]) -> ProbeVerdict | None:
    """The first inline, evaluated, flagged verdict -- the one that caused an abort.

    Only ``execution_mode=inline`` verdicts can gate generation; an async
    probe may also be ``flagged=True`` (it's observational) but must never
    be reported as the reason generation stopped.
    """
    for verdict in verdicts:
        if verdict.execution_mode == ExecutionMode.INLINE and verdict.evaluated and verdict.flagged:
            return verdict
    return None


def build_probing_object(verdicts: Sequence[ProbeVerdict]) -> dict[str, dict[str, Any]]:
    """Build the ``probing`` object shared by clean non-streaming and streaming responses.

    One entry per verdict, keyed by ``extraction_point_name``, each of the
    shape::

        {
            "score": <float | {"max_score": ..., "final_score": ...} | None>,
            "flagged": <bool>,
            "execution_mode": "inline" | "async",
            "evaluated": <bool>,
        }

    Includes every configured probe passed in, inline and async alike --
    this is the "clean" summary object, not filtered to only flagged ones.
    """
    return {
        v.extraction_point_name: {
            "score": v.score_field(),
            "flagged": v.flagged,
            "execution_mode": v.execution_mode.value,
            "evaluated": v.evaluated,
        }
        for v in verdicts
    }


def build_completion_response(
    outcome: GenerationOutcome,
    verdicts: Sequence[ProbeVerdict],
) -> dict[str, Any]:
    """Build the non-streaming, JSON-serializable completion response.

    Two shapes, disambiguated by the top-level ``"flagged"`` key:

    Flagged (an inline probe aborted generation)::

        {
            "flagged": True,
            "finish_reason": "content_flagged",
            "text": None,
            "flagged_by": "toxicity_probe",
            "flagged_score": 0.94,
            "tokens_generated": 17,
            "max_tokens": 256,
            "tokens_saved": 239,
        }

    Clean (generation completed normally)::

        {
            "flagged": False,
            "finish_reason": "stop",
            "text": "...the completion...",
            "tokens_generated": 42,
            "max_tokens": 256,
            "probing": { <extraction_point_name>: {...}, ... },
        }

    Args:
        outcome: the generation outcome (see :class:`GenerationOutcome`).
        verdicts: one :class:`ProbeVerdict` per configured extraction
            point, inline and async alike.

    Returns:
        A plain ``dict`` containing only JSON-serializable values.

    Note:
        Which path is taken is driven by ``outcome.completed``, not by
        re-deriving it from ``verdicts`` -- the caller (the serving layer)
        is the source of truth for whether generation actually stopped
        early. If ``completed`` is ``False`` but no inline verdict is
        flagged (shouldn't happen in practice), ``flagged_by``/
        ``flagged_score`` fall back to ``None``.
    """
    if not outcome.completed:
        flagging_verdict = _find_flagging_verdict(verdicts)
        return {
            "flagged": True,
            "finish_reason": FLAGGED_FINISH_REASON,
            "text": None,
            "flagged_by": flagging_verdict.extraction_point_name if flagging_verdict else None,
            "flagged_score": flagging_verdict.score_field() if flagging_verdict else None,
            "tokens_generated": outcome.tokens_generated,
            "max_tokens": outcome.max_tokens,
            "tokens_saved": outcome.tokens_saved,
        }

    return {
        "flagged": False,
        "finish_reason": "stop",
        "text": outcome.text,
        "tokens_generated": outcome.tokens_generated,
        "max_tokens": outcome.max_tokens,
        "probing": build_probing_object(verdicts),
    }


def build_flagged_sse_chunk(
    outcome: GenerationOutcome,
    verdicts: Sequence[ProbeVerdict],
    *,
    chunk_id: str | None = None,
) -> dict[str, Any]:
    """Build the terminal SSE chunk for a mid-stream flagged abort.

    Intended to be the *last* chunk emitted on the flagged path -- no
    further content-delta chunks should follow it. Shaped like an
    OpenAI-style streaming chunk's terminal frame (a ``choices[0]`` entry
    with ``delta: {}`` and a non-``None`` ``finish_reason``), plus the same
    flagging details as the non-streaming flagged response.

    Args:
        outcome: the generation outcome; only used for
            ``tokens_generated``/``max_tokens``/``tokens_saved`` here (text
            is never included on the flagged path).
        verdicts: the full verdict list for this request; the flagging one
            is located internally via the first inline+evaluated+flagged
            entry.
        chunk_id: optional caller-supplied id (e.g. the same id used on
            earlier chunks of this stream) to echo back in the chunk. Not
            required -- omit if the caller doesn't track one.

    Returns:
        A plain ``dict`` suitable for serializing as ``data: <json>\\n\\n``
        (the caller owns actual SSE framing/serialization).
    """
    flagging_verdict = _find_flagging_verdict(verdicts)
    chunk: dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": FLAGGED_FINISH_REASON,
            }
        ],
        "flagged": True,
        "flagged_by": flagging_verdict.extraction_point_name if flagging_verdict else None,
        "flagged_score": flagging_verdict.score_field() if flagging_verdict else None,
        "tokens_generated": outcome.tokens_generated,
        "max_tokens": outcome.max_tokens,
        "tokens_saved": outcome.tokens_saved,
    }
    if chunk_id is not None:
        chunk["id"] = chunk_id
    return chunk


def build_probing_result_sse_event(
    verdicts: Sequence[ProbeVerdict],
    *,
    chunk_id: str | None = None,
) -> dict[str, Any]:
    """Build the extra SSE event emitted after a clean stream's terminal chunk.

    On the clean streaming path, the normal terminal chunk (``finish_reason:
    "stop"``) is emitted exactly as it would be without probing at all --
    this function builds one *additional* event to send right after it,
    named by :data:`PROBING_RESULT_EVENT`, carrying the same ``probing``
    object as the non-streaming clean response.

    Args:
        verdicts: one :class:`ProbeVerdict` per configured extraction
            point, inline and async alike.
        chunk_id: optional caller-supplied id to echo back, as in
            :func:`build_flagged_sse_chunk`.

    Returns:
        A dict with ``"event"`` (always :data:`PROBING_RESULT_EVENT`) and
        ``"data"`` (the JSON-serializable payload: ``{"probing": {...}}``,
        plus ``"id"`` if ``chunk_id`` was given). The caller is responsible
        for actual SSE wire framing, e.g.::

            event = build_probing_result_sse_event(verdicts)
            f"event: {event['event']}\\ndata: {json.dumps(event['data'])}\\n\\n"
    """
    data: dict[str, Any] = {"probing": build_probing_object(verdicts)}
    if chunk_id is not None:
        data["id"] = chunk_id
    return {"event": PROBING_RESULT_EVENT, "data": data}

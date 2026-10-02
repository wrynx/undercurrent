"""Unit tests for the ``examples/openai_server/response_schema.py`` reference example.

Pure formatting logic -- no vLLM, no HTTP, no router/probe machinery
involved. Constructs GenerationOutcome/ProbeVerdict values directly and
checks the resulting dicts.
"""

from __future__ import annotations

import pytest
from response_schema import (
    FLAGGED_FINISH_REASON,
    PROBING_RESULT_EVENT,
    GenerationOutcome,
    ProbeVerdict,
    TrajectoryAggregate,
    aggregate_trajectory_scores,
    build_completion_response,
    build_flagged_sse_chunk,
    build_probing_object,
    build_probing_result_sse_event,
)

from undercurrent.spec import ExecutionMode, ProbeKind

# ---------------------------------------------------------------------------
# Trajectory aggregation
# ---------------------------------------------------------------------------


def test_aggregate_trajectory_scores_max_and_final():
    scores = [0.1, 0.7, 0.3, 0.5]
    agg = aggregate_trajectory_scores(scores)
    assert agg == TrajectoryAggregate(max_score=0.7, final_score=0.5)


def test_aggregate_trajectory_scores_single_value():
    agg = aggregate_trajectory_scores([0.42])
    assert agg.max_score == 0.42
    assert agg.final_score == 0.42


def test_aggregate_trajectory_scores_empty_raises():
    with pytest.raises(ValueError):
        aggregate_trajectory_scores([])


def test_trajectory_aggregate_as_dict():
    agg = TrajectoryAggregate(max_score=0.9, final_score=0.2)
    assert agg.as_dict() == {"max_score": 0.9, "final_score": 0.2}


# ---------------------------------------------------------------------------
# Clean, non-streaming
# ---------------------------------------------------------------------------


def test_clean_non_streaming_multiple_probes_mixed_modes():
    outcome = GenerationOutcome(
        completed=True,
        text="hello world",
        tokens_generated=42,
        max_tokens=256,
    )
    verdicts = [
        ProbeVerdict(
            extraction_point_name="toxicity_probe",
            execution_mode=ExecutionMode.INLINE,
            probe_kind=ProbeKind.SINGLE_SHOT,
            flagged=False,
            score=0.12,
            evaluated=True,
        ),
        ProbeVerdict(
            extraction_point_name="trajectory_probe",
            execution_mode=ExecutionMode.ASYNC,
            probe_kind=ProbeKind.TRAJECTORY,
            flagged=False,
            score=aggregate_trajectory_scores([0.1, 0.4, 0.2]),
            evaluated=True,
        ),
        ProbeVerdict(
            # never reached, e.g. generated[*] probe with no generated tokens
            extraction_point_name="unreached_async_probe",
            execution_mode=ExecutionMode.ASYNC,
            probe_kind=ProbeKind.TRAJECTORY,
            flagged=False,
            score=None,
            evaluated=False,
        ),
    ]

    response = build_completion_response(outcome, verdicts)

    assert response["flagged"] is False
    assert response["finish_reason"] == "stop"
    assert response["text"] == "hello world"
    assert response["tokens_generated"] == 42
    assert response["max_tokens"] == 256
    assert "flagged_by" not in response

    probing = response["probing"]
    assert set(probing.keys()) == {"toxicity_probe", "trajectory_probe", "unreached_async_probe"}

    assert probing["toxicity_probe"] == {
        "score": 0.12,
        "flagged": False,
        "execution_mode": "inline",
        "evaluated": True,
    }
    assert probing["trajectory_probe"] == {
        "score": {"max_score": 0.4, "final_score": 0.2},
        "flagged": False,
        "execution_mode": "async",
        "evaluated": True,
    }
    assert probing["unreached_async_probe"] == {
        "score": None,
        "flagged": False,
        "execution_mode": "async",
        "evaluated": False,
    }


def test_build_probing_object_standalone():
    verdicts = [
        ProbeVerdict(
            extraction_point_name="p1",
            execution_mode=ExecutionMode.INLINE,
            probe_kind=ProbeKind.SINGLE_SHOT,
            flagged=False,
            score=0.5,
            evaluated=True,
        ),
    ]
    assert build_probing_object(verdicts) == {
        "p1": {"score": 0.5, "flagged": False, "execution_mode": "inline", "evaluated": True},
    }


# ---------------------------------------------------------------------------
# Flagged, non-streaming
# ---------------------------------------------------------------------------


def test_flagged_non_streaming_inline_probe_fired():
    outcome = GenerationOutcome(
        completed=False,
        text=None,
        tokens_generated=17,
        max_tokens=256,
        aborted_reason="toxicity_probe",
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
        ProbeVerdict(
            extraction_point_name="other_async_probe",
            execution_mode=ExecutionMode.ASYNC,
            probe_kind=ProbeKind.TRAJECTORY,
            flagged=False,
            score=None,
            evaluated=False,
        ),
    ]

    response = build_completion_response(outcome, verdicts)

    assert response == {
        "flagged": True,
        "finish_reason": FLAGGED_FINISH_REASON,
        "text": None,
        "flagged_by": "toxicity_probe",
        "flagged_score": 0.94,
        "tokens_generated": 17,
        "max_tokens": 256,
        "tokens_saved": 239,
    }
    # Clean-only fields must not leak into the flagged shape.
    assert "probing" not in response


def test_flagged_non_streaming_ignores_async_flag_as_the_cause():
    """An async probe being flagged must never be reported as the abort cause."""
    outcome = GenerationOutcome(completed=False, text=None, tokens_generated=5, max_tokens=100)
    verdicts = [
        ProbeVerdict(
            extraction_point_name="async_probe",
            execution_mode=ExecutionMode.ASYNC,
            probe_kind=ProbeKind.TRAJECTORY,
            flagged=True,
            score=0.99,
            evaluated=True,
        ),
        ProbeVerdict(
            extraction_point_name="inline_probe",
            execution_mode=ExecutionMode.INLINE,
            probe_kind=ProbeKind.SINGLE_SHOT,
            flagged=True,
            score=0.8,
            evaluated=True,
        ),
    ]

    response = build_completion_response(outcome, verdicts)
    assert response["flagged_by"] == "inline_probe"
    assert response["flagged_score"] == 0.8


def test_tokens_saved_clamped_at_zero():
    outcome = GenerationOutcome(completed=False, text=None, tokens_generated=300, max_tokens=256)
    assert outcome.tokens_saved == 0


# ---------------------------------------------------------------------------
# Clean streaming
# ---------------------------------------------------------------------------


def test_clean_streaming_probing_result_event():
    verdicts = [
        ProbeVerdict(
            extraction_point_name="toxicity_probe",
            execution_mode=ExecutionMode.INLINE,
            probe_kind=ProbeKind.SINGLE_SHOT,
            flagged=False,
            score=0.05,
            evaluated=True,
        ),
        ProbeVerdict(
            extraction_point_name="trajectory_probe",
            execution_mode=ExecutionMode.ASYNC,
            probe_kind=ProbeKind.TRAJECTORY,
            flagged=False,
            score=aggregate_trajectory_scores([0.2, 0.9, 0.1]),
            evaluated=True,
        ),
    ]

    event = build_probing_result_sse_event(verdicts, chunk_id="chatcmpl-123")

    assert event["event"] == PROBING_RESULT_EVENT
    assert event["data"]["id"] == "chatcmpl-123"
    assert event["data"]["probing"] == {
        "toxicity_probe": {
            "score": 0.05,
            "flagged": False,
            "execution_mode": "inline",
            "evaluated": True,
        },
        "trajectory_probe": {
            "score": {"max_score": 0.9, "final_score": 0.1},
            "flagged": False,
            "execution_mode": "async",
            "evaluated": True,
        },
    }


def test_clean_streaming_event_omits_id_when_not_given():
    event = build_probing_result_sse_event([])
    assert "id" not in event["data"]
    assert event["data"]["probing"] == {}


# ---------------------------------------------------------------------------
# Flagged streaming
# ---------------------------------------------------------------------------


def test_flagged_streaming_early_stop_chunk():
    outcome = GenerationOutcome(completed=False, text=None, tokens_generated=10, max_tokens=200)
    verdicts = [
        ProbeVerdict(
            extraction_point_name="toxicity_probe",
            execution_mode=ExecutionMode.INLINE,
            probe_kind=ProbeKind.SINGLE_SHOT,
            flagged=True,
            score=0.88,
            evaluated=True,
        ),
    ]

    chunk = build_flagged_sse_chunk(outcome, verdicts, chunk_id="chatcmpl-456")

    assert chunk["id"] == "chatcmpl-456"
    assert chunk["choices"] == [{"index": 0, "delta": {}, "finish_reason": FLAGGED_FINISH_REASON}]
    assert chunk["flagged"] is True
    assert chunk["flagged_by"] == "toxicity_probe"
    assert chunk["flagged_score"] == 0.88
    assert chunk["tokens_generated"] == 10
    assert chunk["max_tokens"] == 200
    assert chunk["tokens_saved"] == 190


def test_flagged_streaming_chunk_without_id():
    outcome = GenerationOutcome(completed=False, text=None, tokens_generated=1, max_tokens=50)
    verdicts = [
        ProbeVerdict(
            extraction_point_name="p",
            execution_mode=ExecutionMode.INLINE,
            probe_kind=ProbeKind.SINGLE_SHOT,
            flagged=True,
            score=None,
            evaluated=True,
        ),
    ]
    chunk = build_flagged_sse_chunk(outcome, verdicts)
    assert "id" not in chunk
    assert chunk["flagged_score"] is None


def test_flagged_streaming_no_flagging_verdict_found_defaults_to_none():
    """Defensive path: completed=False but nothing in verdicts is flagged inline."""
    outcome = GenerationOutcome(completed=False, text=None, tokens_generated=1, max_tokens=50)
    chunk = build_flagged_sse_chunk(outcome, verdicts=[])
    assert chunk["flagged_by"] is None
    assert chunk["flagged_score"] is None

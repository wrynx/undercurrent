"""Unit tests for the seq_id -> (request_id, token_pos) translation core.

Deliberately pure Python: every "step" here is a hand-built
`StepBatchMetadata`, standing in for what `introspection.py` would have read
off a live vLLM `ModelRunner`. No vLLM, no torch, no GPU -- this is the
"build and unit-test this mapping in isolation before wiring it into live
generation" step the task calls for.
"""

import pytest

from undercurrent.adapters.vllm.seq_mapper import SeqIdMapper, SeqMapperError, StepBatchMetadata


def test_step_batch_metadata_row_range_single_request():
    batch = StepBatchMetadata(req_ids=("a",), num_scheduled_tokens=(5,), num_computed_tokens_before_step=(0,))
    assert batch.row_range("a") == (0, 5)
    assert batch.total_rows == 5


def test_step_batch_metadata_row_range_multi_request():
    batch = StepBatchMetadata(
        req_ids=("a", "b", "c"),
        num_scheduled_tokens=(3, 1, 2),
        num_computed_tokens_before_step=(0, 7, 2),
    )
    assert batch.row_range("a") == (0, 3)
    assert batch.row_range("b") == (3, 4)
    assert batch.row_range("c") == (4, 6)
    assert batch.total_rows == 6


def test_step_batch_metadata_rejects_mismatched_lengths():
    with pytest.raises(SeqMapperError):
        StepBatchMetadata(req_ids=("a", "b"), num_scheduled_tokens=(1,), num_computed_tokens_before_step=(0, 0))


def test_step_batch_metadata_rejects_duplicate_req_ids():
    with pytest.raises(SeqMapperError):
        StepBatchMetadata(req_ids=("a", "a"), num_scheduled_tokens=(1, 1), num_computed_tokens_before_step=(0, 1))


def test_step_batch_metadata_rejects_zero_scheduled_tokens():
    with pytest.raises(SeqMapperError):
        StepBatchMetadata(req_ids=("a",), num_scheduled_tokens=(0,), num_computed_tokens_before_step=(0,))


def test_step_batch_metadata_rejects_negative_computed_tokens():
    with pytest.raises(SeqMapperError):
        StepBatchMetadata(req_ids=("a",), num_scheduled_tokens=(1,), num_computed_tokens_before_step=(-1,))


def test_register_request_rejects_zero_prompt_len():
    mapper = SeqIdMapper()
    with pytest.raises(SeqMapperError):
        mapper.register_request("req-1", prompt_len=0)


def test_register_request_rejects_duplicate():
    mapper = SeqIdMapper()
    mapper.register_request("req-1", prompt_len=4)
    with pytest.raises(SeqMapperError):
        mapper.register_request("req-1", prompt_len=4)


def test_resolve_step_rejects_unregistered_request():
    mapper = SeqIdMapper()
    batch = StepBatchMetadata(req_ids=("ghost",), num_scheduled_tokens=(1,), num_computed_tokens_before_step=(0,))
    with pytest.raises(SeqMapperError):
        list(mapper.resolve_step(batch))


def test_single_shot_prefill_step_all_prompt_tokens():
    """A one-shot (unchunked) prefill: the whole prompt arrives in one step."""
    mapper = SeqIdMapper()
    mapper.register_request("req-1", prompt_len=5)
    batch = StepBatchMetadata(req_ids=("req-1",), num_scheduled_tokens=(5,), num_computed_tokens_before_step=(0,))
    mappings = list(mapper.resolve_step(batch))

    assert [m.token_index for m in mappings] == [0, 1, 2, 3, 4]
    assert all(not m.is_generated for m in mappings)
    assert all(m.generated_index is None for m in mappings)
    assert [m.row for m in mappings] == [0, 1, 2, 3, 4]
    assert mapper.num_generated_so_far("req-1") == 0


def test_chunked_prefill_spans_multiple_steps():
    """A long prompt prefilled across two steps must still produce
    contiguous, correctly-offset absolute token_index values."""
    mapper = SeqIdMapper()
    mapper.register_request("req-1", prompt_len=10)

    step1 = StepBatchMetadata(req_ids=("req-1",), num_scheduled_tokens=(6,), num_computed_tokens_before_step=(0,))
    mappings1 = list(mapper.resolve_step(step1))
    assert [m.token_index for m in mappings1] == [0, 1, 2, 3, 4, 5]
    assert all(not m.is_generated for m in mappings1)

    step2 = StepBatchMetadata(req_ids=("req-1",), num_scheduled_tokens=(4,), num_computed_tokens_before_step=(6,))
    mappings2 = list(mapper.resolve_step(step2))
    assert [m.token_index for m in mappings2] == [6, 7, 8, 9]
    assert all(not m.is_generated for m in mappings2)
    assert mapper.num_generated_so_far("req-1") == 0


def test_decode_steps_advance_generated_index_one_token_at_a_time():
    mapper = SeqIdMapper()
    mapper.register_request("req-1", prompt_len=3)
    # prefill
    list(mapper.resolve_step(StepBatchMetadata(("req-1",), (3,), (0,))))

    for expected_token_index, expected_gen_index in [(3, 0), (4, 1), (5, 2)]:
        n_before = expected_token_index
        step = StepBatchMetadata(
            req_ids=("req-1",), num_scheduled_tokens=(1,), num_computed_tokens_before_step=(n_before,)
        )
        mappings = list(mapper.resolve_step(step))
        assert len(mappings) == 1
        m = mappings[0]
        assert m.token_index == expected_token_index
        assert m.is_generated is True
        assert m.generated_index == expected_gen_index

    assert mapper.num_generated_so_far("req-1") == 3


def test_two_concurrent_requests_share_one_step_with_distinct_row_ranges():
    """The core continuous-batching case: two different requests, at two
    different absolute positions in their own sequences, occupy disjoint row
    ranges of the SAME flattened step tensor."""
    mapper = SeqIdMapper()
    mapper.register_request("req-a", prompt_len=4)
    mapper.register_request("req-b", prompt_len=2)

    # req-a is mid-decode (4 prompt tokens + 2 already generated -> next token is index 6);
    # req-b is still finishing prefill in the same step.
    batch = StepBatchMetadata(
        req_ids=("req-a", "req-b"),
        num_scheduled_tokens=(1, 2),
        num_computed_tokens_before_step=(6, 0),
    )
    mappings = list(mapper.resolve_step(batch))
    by_request = {(m.request_id, m.row): m for m in mappings}

    a = by_request[("req-a", 0)]
    assert a.token_index == 6
    assert a.is_generated is True
    assert a.generated_index == 2

    b0 = by_request[("req-b", 1)]
    b1 = by_request[("req-b", 2)]
    assert (b0.token_index, b0.is_generated) == (0, False)
    assert (b1.token_index, b1.is_generated) == (1, False)


def test_finished_requests_row_slot_reused_by_a_new_request_next_step():
    """vLLM reuses a finished request's persistent-batch row slot for a
    newly admitted request. The mapper must not let that reuse leak state
    between the two request_ids -- it's keyed by request_id, not row."""
    mapper = SeqIdMapper()
    mapper.register_request("req-old", prompt_len=2)
    step1 = StepBatchMetadata(req_ids=("req-old",), num_scheduled_tokens=(2,), num_computed_tokens_before_step=(0,))
    list(mapper.resolve_step(step1))
    # req-old finishes and is torn down.
    mapper.unregister_request("req-old")

    mapper.register_request("req-new", prompt_len=3)
    # req-new lands in the same row slot (row 0) that req-old occupied.
    step2 = StepBatchMetadata(req_ids=("req-new",), num_scheduled_tokens=(3,), num_computed_tokens_before_step=(0,))
    mappings = list(mapper.resolve_step(step2))

    assert [m.token_index for m in mappings] == [0, 1, 2]
    assert all(m.request_id == "req-new" for m in mappings)
    assert not mapper.is_registered("req-old")

    # Old request_id is gone; asking about it is an error, not stale data.
    with pytest.raises(SeqMapperError):
        mapper.num_generated_so_far("req-old")


def test_request_dropping_out_of_batch_for_a_step_is_not_an_error():
    """A request resident in the persistent batch but not scheduled this
    step (e.g. temporarily starved) simply contributes no rows -- callers
    (introspection.py) are expected to omit it from that step's
    StepBatchMetadata entirely; the mapper doesn't need to do anything
    special, it just never sees it that step."""
    mapper = SeqIdMapper()
    mapper.register_request("req-a", prompt_len=2)
    mapper.register_request("req-b", prompt_len=2)

    step1 = StepBatchMetadata(("req-a", "req-b"), (2, 2), (0, 0))
    list(mapper.resolve_step(step1))

    # req-b is omitted this step (not scheduled) -- only req-a appears.
    step2 = StepBatchMetadata(("req-a",), (1,), (2,))
    mappings = list(mapper.resolve_step(step2))
    assert [m.request_id for m in mappings] == ["req-a"]
    # req-b's counters are untouched.
    assert mapper.num_generated_so_far("req-b") == 0

    # req-b resumes on a later step, continuing from where it left off.
    step3 = StepBatchMetadata(("req-b",), (1,), (2,))
    mappings3 = list(mapper.resolve_step(step3))
    assert mappings3[0].token_index == 2
    assert mappings3[0].is_generated is True


def test_prompt_len_accessor():
    mapper = SeqIdMapper()
    mapper.register_request("req-1", prompt_len=7)
    assert mapper.prompt_len("req-1") == 7
    with pytest.raises(SeqMapperError):
        mapper.prompt_len("no-such-request")


def test_registered_request_ids_reflects_register_and_unregister():
    mapper = SeqIdMapper()
    assert mapper.registered_request_ids() == frozenset()
    mapper.register_request("req-1", prompt_len=1)
    mapper.register_request("req-2", prompt_len=1)
    assert mapper.registered_request_ids() == frozenset({"req-1", "req-2"})
    mapper.unregister_request("req-1")
    assert mapper.registered_request_ids() == frozenset({"req-2"})
    # Unregistering an already-gone (or never-registered) request is a no-op, not an error.
    mapper.unregister_request("req-1")
    mapper.unregister_request("never-existed")

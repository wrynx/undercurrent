import pytest

from undercurrent.spec import PositionSyntaxError
from undercurrent.spec.position import PositionKind, PositionSelector, parse_position


def test_absolute_int():
    sel = parse_position(5)
    assert sel.kind == PositionKind.ABSOLUTE
    assert sel.index == 5
    assert sel.matches(5, is_generated=False, prompt_len=10)
    assert not sel.matches(4, is_generated=False, prompt_len=10)


def test_absolute_str():
    sel = parse_position("7")
    assert sel.kind == PositionKind.ABSOLUTE
    assert sel.index == 7


def test_prompt_absolute_index():
    sel = parse_position("prompt[0]")
    assert sel.kind == PositionKind.PROMPT_INDEX
    assert sel.matches(0, is_generated=False, prompt_len=10)
    assert not sel.matches(1, is_generated=False, prompt_len=10)


def test_prompt_negative_index():
    sel = parse_position("prompt[-1]")
    assert sel.kind == PositionKind.PROMPT_INDEX
    # last prompt token in a 10-token prompt is absolute index 9
    assert sel.matches(9, is_generated=False, prompt_len=10)
    assert not sel.matches(8, is_generated=False, prompt_len=10)


def test_generated_positive_index():
    sel = parse_position("generated[0]")
    assert sel.kind == PositionKind.GENERATED_INDEX
    # first generated token after a 10-token prompt is absolute index 10
    assert sel.matches(10, is_generated=True, prompt_len=10, generated_index=0)
    assert not sel.matches(11, is_generated=True, prompt_len=10, generated_index=1)


def test_generated_negative_index_requires_total():
    sel = parse_position("generated[-1]")
    assert sel.kind == PositionKind.GENERATED_INDEX
    # Cannot resolve without knowing total generated length.
    assert not sel.matches(15, is_generated=True, prompt_len=10, generated_index=5)
    # Once total is known (say, 6 generated tokens, last is generated_index=5,
    # absolute index=15), it resolves.
    assert sel.matches(15, is_generated=True, prompt_len=10, generated_index=5, num_generated_total=6)
    assert not sel.matches(14, is_generated=True, prompt_len=10, generated_index=4, num_generated_total=6)


def test_generated_wildcard():
    sel = parse_position("generated[*]")
    assert sel.kind == PositionKind.GENERATED_WILDCARD
    assert sel.is_continuous
    assert sel.matches(10, is_generated=True, prompt_len=10, generated_index=0)
    assert sel.matches(20, is_generated=True, prompt_len=10, generated_index=10)
    assert not sel.matches(5, is_generated=False, prompt_len=10)


def test_generated_slice_open_ended():
    sel = parse_position("generated[5:]")
    assert sel.kind == PositionKind.GENERATED_SLICE
    assert sel.is_continuous
    assert sel.slice_start == 5
    assert sel.slice_stop is None
    assert not sel.matches(14, is_generated=True, prompt_len=10, generated_index=4)
    assert sel.matches(15, is_generated=True, prompt_len=10, generated_index=5)
    assert sel.matches(25, is_generated=True, prompt_len=10, generated_index=15)


def test_generated_slice_bounded():
    sel = parse_position("generated[2:8]")
    assert sel.slice_start == 2
    assert sel.slice_stop == 8
    assert not sel.matches(11, is_generated=True, prompt_len=10, generated_index=1)
    assert sel.matches(12, is_generated=True, prompt_len=10, generated_index=2)
    assert sel.matches(17, is_generated=True, prompt_len=10, generated_index=7)
    assert not sel.matches(18, is_generated=True, prompt_len=10, generated_index=8)


def test_generated_slice_invalid_range():
    with pytest.raises(PositionSyntaxError):
        parse_position("generated[8:2]")


def test_offset_modifier_prompt():
    # "read position b at token b+1": base = prompt[-1] (index 9 in a
    # 10-token prompt), offset +1 -> absolute index 10 (first generated token).
    sel = parse_position("prompt[-1]+1")
    assert sel.offset == 1
    assert sel.matches(10, is_generated=True, prompt_len=10, generated_index=0)
    assert not sel.matches(9, is_generated=False, prompt_len=10)


def test_offset_modifier_generated_negative_offset():
    sel = parse_position("generated[5]-2")
    assert sel.offset == -2
    # generated[5] with a 10-token prompt is absolute 15; -2 -> 13
    assert sel.matches(13, is_generated=True, prompt_len=10, generated_index=3)


@pytest.mark.parametrize(
    "bad_value",
    [
        "prompt[]",
        "generated[]",
        "generated[*",
        "generated[5:2]",
        "foo[1]",
        "prompt[1.5]",
        "",
        "   ",
        None,
        1.5,
        True,
    ],
)
def test_invalid_positions_raise(bad_value):
    with pytest.raises(PositionSyntaxError):
        parse_position(bad_value)


def test_to_raw_round_trips_through_parse():
    for raw in ["5", "prompt[-1]", "generated[0]", "generated[*]", "generated[5:]", "generated[2:8]", "prompt[-1]+1"]:
        sel = parse_position(raw)
        reparsed = parse_position(sel.to_raw())
        assert reparsed == sel


@pytest.mark.parametrize("kind", [PositionKind.ABSOLUTE, PositionKind.PROMPT_INDEX, PositionKind.GENERATED_INDEX])
def test_point_selector_without_index_raises_clear_error(kind):
    # Only reachable by constructing PositionSelector directly; parse_position
    # always sets `index` for these kinds. Previously a TypeError on None.
    selector = PositionSelector(kind=kind)
    with pytest.raises(ValueError, match="has no index"):
        selector.resolve_absolute(prompt_len=4, num_generated_total=2)

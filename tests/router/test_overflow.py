import threading
import time

import pytest

from undercurrent.router.overflow import EMPTY, BoundedDropQueue, OverflowPolicy


def test_put_get_basic_fifo_order():
    q = BoundedDropQueue(maxsize=3, policy=OverflowPolicy.DROP_OLDEST)
    q.put(1)
    q.put(2)
    q.put(3)
    assert q.get() == 1
    assert q.get() == 2
    assert q.get() == 3


def test_drop_oldest_evicts_earliest_item():
    q = BoundedDropQueue(maxsize=2, policy=OverflowPolicy.DROP_OLDEST)
    q.put("a")
    q.put("b")
    q.put("c")  # queue full at put time -> evicts "a"
    assert len(q) == 2
    assert q.get() == "b"
    assert q.get() == "c"


def test_drop_newest_discards_incoming_item():
    q = BoundedDropQueue(maxsize=2, policy=OverflowPolicy.DROP_NEWEST)
    assert q.put("a") is True
    assert q.put("b") is True
    assert q.put("c") is False  # discarded, queue unchanged
    assert len(q) == 2
    assert q.get() == "a"
    assert q.get() == "b"


def test_block_policy_blocks_until_space_frees():
    q = BoundedDropQueue(maxsize=1, policy=OverflowPolicy.BLOCK)
    q.put("a")

    unblocked = threading.Event()

    def _producer():
        q.put("b")  # should block until "a" is consumed
        unblocked.set()

    t = threading.Thread(target=_producer)
    t.start()

    time.sleep(0.1)
    assert not unblocked.is_set()  # still blocked

    assert q.get() == "a"  # frees a slot
    t.join(timeout=2)
    assert unblocked.is_set()
    assert q.get() == "b"


def test_get_returns_empty_sentinel_on_timeout():
    q = BoundedDropQueue(maxsize=1, policy=OverflowPolicy.DROP_OLDEST)
    assert q.get(timeout=0.05) is EMPTY


def test_drain_removes_and_returns_everything_without_blocking():
    q = BoundedDropQueue(maxsize=5, policy=OverflowPolicy.DROP_OLDEST)
    q.put(1)
    q.put(2)
    q.put(3)
    items = q.drain()
    assert items == [1, 2, 3]
    assert len(q) == 0


def test_close_wakes_blocked_get_and_returns_empty_when_drained():
    q = BoundedDropQueue(maxsize=1, policy=OverflowPolicy.DROP_OLDEST)
    result = {}

    def _consumer():
        result["value"] = q.get(timeout=5)

    t = threading.Thread(target=_consumer)
    t.start()
    time.sleep(0.05)
    q.close()
    t.join(timeout=2)
    assert result["value"] is EMPTY


def test_close_still_allows_draining_queued_items():
    q = BoundedDropQueue(maxsize=5, policy=OverflowPolicy.DROP_OLDEST)
    q.put(1)
    q.put(2)
    q.close()
    assert q.get() == 1
    assert q.get() == 2
    assert q.get() is EMPTY


def test_put_after_close_is_rejected():
    q = BoundedDropQueue(maxsize=5, policy=OverflowPolicy.DROP_OLDEST)
    q.close()
    assert q.put(1) is False
    assert len(q) == 0


def test_invalid_maxsize_rejected():
    with pytest.raises(ValueError):
        BoundedDropQueue(maxsize=0, policy=OverflowPolicy.DROP_OLDEST)

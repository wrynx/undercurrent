"""Bounded per-binding async queue with a configurable overflow policy.

Not built on `queue.Queue` because none of Python's stdlib queue types
support "make room by evicting the oldest queued item" -- that's the crux
of drop_oldest, so this implements its own locking directly instead of
layering eviction logic on top of a type that doesn't expose it.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any

from ..errors import ProbingValueError


class OverflowPolicy(str, Enum):
    """What an async extraction point's bounded queue does when it is full."""

    DROP_OLDEST = "drop_oldest"
    """Evict the oldest queued activation to make room (the default)."""
    DROP_NEWEST = "drop_newest"
    """Drop the incoming activation."""
    BLOCK = "block"
    """Block ``route()`` until there is room. Applies backpressure to generation."""


class _Empty:
    """Sentinel distinguishing 'queue is empty' from a real queued `None`."""

    def __repr__(self) -> str:
        return "<EMPTY>"


EMPTY = _Empty()


class BoundedDropQueue:
    """Thread-safe bounded FIFO queue enforcing `maxsize` via `policy`.

    - DROP_OLDEST: evict the oldest queued item to make room for the new one.
    - DROP_NEWEST: discard the incoming item; the queue is left unchanged.
    - BLOCK: `put()` blocks until space frees up (backpressure on the caller).
    """

    def __init__(
        self,
        maxsize: int,
        policy: OverflowPolicy,
        on_drop: Callable[[], None] | None = None,
    ) -> None:
        if maxsize < 1:
            raise ProbingValueError(
                f"queue depth must be >= 1 (got {maxsize!r}). Set the extraction point's queue_depth, or "
                "Router(default_queue_depth=...), to a positive int."
            )
        self._maxsize = maxsize
        self._policy = policy
        self._items: deque[Any] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False
        # Called synchronously, still holding the queue lock, exactly when
        # an item is discarded by the overflow policy itself (an eviction
        # under DROP_OLDEST, a rejection under DROP_NEWEST) -- not when
        # put() is merely rejected because the queue is already closed,
        # which is shutdown behavior rather than an overflow drop. Held
        # under the lock so the count this drives (e.g. a metrics
        # registry's drop_count) can never race with a concurrent put()
        # and under/over-count; the callback itself must therefore be fast
        # and must never call back into this queue.
        self._on_drop = on_drop

    def put(self, item: Any) -> bool:
        """Enqueue `item`.

        Returns True if it was accepted, False if it was dropped (only
        possible under DROP_NEWEST, or if the queue has been closed).
        DROP_OLDEST always accepts by evicting; BLOCK always accepts by
        waiting, unless the queue is closed while it's waiting.
        """
        with self._not_full:
            if self._closed:
                return False
            if len(self._items) >= self._maxsize:
                if self._policy == OverflowPolicy.DROP_OLDEST:
                    self._items.popleft()
                    if self._on_drop is not None:
                        self._on_drop()
                elif self._policy == OverflowPolicy.DROP_NEWEST:
                    if self._on_drop is not None:
                        self._on_drop()
                    return False
                else:  # BLOCK
                    while len(self._items) >= self._maxsize and not self._closed:
                        self._not_full.wait()
                    if self._closed:
                        return False
            self._items.append(item)
            self._not_empty.notify()
            return True

    def get(self, timeout: float | None = None) -> Any:
        """Block up to `timeout` seconds (or forever, if None) for an item.

        Returns the item, or the `EMPTY` sentinel if the wait timed out, or
        if the queue was closed and is now fully drained.
        """
        with self._not_empty:
            while not self._items and not self._closed:
                if not self._not_empty.wait(timeout=timeout):
                    return EMPTY
            if self._items:
                item = self._items.popleft()
                self._not_full.notify()
                return item
            return EMPTY

    def drain(self) -> list[Any]:
        """Remove and return everything currently queued, without blocking."""
        with self._lock:
            items = list(self._items)
            self._items.clear()
            self._not_full.notify_all()
            return items

    def close(self) -> None:
        """Mark closed: wakes any blocked put()/get() calls.

        Items already queued are still returned by subsequent get() calls
        (draining continues to work after close); further put() calls are
        rejected.
        """
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def close_and_discard(self) -> None:
        """Mark closed and clear whatever is currently queued, atomically
        under one lock acquisition.

        Unlike calling `close()` then `drain()` as two separate calls,
        this can't race a concurrently-woken `get()` in a consumer thread
        stealing one item between the two steps -- a blocked `get()` can
        only observe the queue either before this runs (sees whatever was
        queued, same as any other close) or after (sees closed and empty,
        returns EMPTY immediately). Used for an immediate, non-draining
        cancel (see `binding.AsyncWorker.cancel`).
        """
        with self._lock:
            self._closed = True
            self._items.clear()
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

import json
import threading
import time

from undercurrent.core import ProbeAction, ProbeSignal
from undercurrent.sinks import WebhookLogSink


class _CountingPostFn:
    """Test double standing in for the real HTTP POST: fails a fixed
    number of times before succeeding (or always fails), and records every
    attempt -- lets tests assert exact retry counts without a real network
    call or server."""

    def __init__(self, fail_times=0, always_fail=False):
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, url, record):
        with self._lock:
            self.calls.append((url, record))
            call_index = len(self.calls)
        if self.always_fail or call_index <= self.fail_times:
            raise ConnectionError("simulated webhook failure")


def test_successful_post_delivers_without_retry():
    post_fn = _CountingPostFn(fail_times=0)
    sink = WebhookLogSink("https://example.invalid/hook", post_fn=post_fn, backoff_base=0.01)

    sink.write_signal("req-1", "ep-1", ProbeSignal(action=ProbeAction.CONTINUE))
    sink.close(timeout=2)

    assert len(post_fn.calls) == 1
    url, record = post_fn.calls[0]
    assert url == "https://example.invalid/hook"
    assert record["kind"] == "signal"
    assert record["request_id"] == "req-1"


def test_retries_on_simulated_failure_then_succeeds():
    post_fn = _CountingPostFn(fail_times=2)  # fails attempts 1-2, succeeds on attempt 3
    sink = WebhookLogSink("https://example.invalid/hook", post_fn=post_fn, max_retries=2, backoff_base=0.01)

    sink.write_signal("req-1", "ep-1", ProbeSignal())
    sink.close(timeout=2)

    assert len(post_fn.calls) == 3  # 1 initial attempt + 2 retries


def test_exhausted_retries_fall_back_to_dead_letter_file(tmp_path):
    dead_letter_path = tmp_path / "dead_letters.ndjson"
    post_fn = _CountingPostFn(always_fail=True)
    sink = WebhookLogSink(
        "https://example.invalid/hook",
        post_fn=post_fn,
        max_retries=2,
        backoff_base=0.01,
        dead_letter_path=dead_letter_path,
    )

    signal = ProbeSignal(action=ProbeAction.ABORT, metadata={"reason": "test"})
    sink.write_signal("req-1", "ep-1", signal)
    sink.close(timeout=2)

    assert len(post_fn.calls) == 3  # 1 initial attempt + 2 retries, all failed
    lines = dead_letter_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["request_id"] == "req-1"
    assert record["payload"]["action"] == "abort"
    assert record["payload"]["metadata"] == {"reason": "test"}


def test_exhausted_retries_without_dead_letter_path_log_locally_via_logging(caplog):
    post_fn = _CountingPostFn(always_fail=True)
    sink = WebhookLogSink("https://example.invalid/hook", post_fn=post_fn, max_retries=1, backoff_base=0.01)

    with caplog.at_level("ERROR", logger="undercurrent.sinks.webhook_sink"):
        sink.write_signal("req-1", "ep-1", ProbeSignal())
        sink.close(timeout=2)

    assert any("permanent delivery failure" in record.message for record in caplog.records)


def test_write_signal_never_raises_and_returns_immediately_even_when_post_always_fails():
    post_fn = _CountingPostFn(always_fail=True)
    sink = WebhookLogSink("https://example.invalid/hook", post_fn=post_fn, max_retries=2, backoff_base=0.5)

    start = time.monotonic()
    sink.write_signal("req-1", "ep-1", ProbeSignal())  # must not raise
    elapsed = time.monotonic() - start

    assert elapsed < 0.1  # enqueue only -- retries/backoff happen on the background thread
    sink.close(timeout=5)


def test_queue_full_drops_and_counts_rather_than_blocking():
    # post_fn blocks forever so the worker never drains the queue, forcing genuine overflow.
    gate = threading.Event()

    def _blocking_post(url, record):
        gate.wait(timeout=5)

    sink = WebhookLogSink("https://example.invalid/hook", post_fn=_blocking_post, queue_maxsize=1)

    sink.write_signal("req-1", "ep-1", ProbeSignal())  # picked up by the worker immediately, blocks it
    time.sleep(0.05)  # give the worker thread time to dequeue it, so the queue below starts empty

    sink.write_signal("req-1", "ep-1", ProbeSignal())  # fills the queue (maxsize=1)
    start = time.monotonic()
    sink.write_signal("req-1", "ep-1", ProbeSignal())  # must be dropped, not block
    elapsed = time.monotonic() - start

    assert elapsed < 0.1
    assert sink.dropped_count == 1

    gate.set()
    sink.close(timeout=2)


def test_unwritable_dead_letter_path_does_not_kill_the_sender(tmp_path, caplog):
    # Regression: a failing dead-letter write used to raise out of the sender
    # thread, so later records were never sent and close() could hang.
    dead_letter_path = tmp_path / "is-a-directory"
    dead_letter_path.mkdir()  # open(..., "a") on it raises IsADirectoryError / PermissionError

    class _FailFirstRecord:
        def __init__(self):
            self.delivered = []

        def __call__(self, url, record):
            if record["request_id"] == "req-fail":
                raise ConnectionError("simulated webhook failure")
            self.delivered.append(record["request_id"])

    post_fn = _FailFirstRecord()
    sink = WebhookLogSink(
        "https://example.invalid/hook",
        post_fn=post_fn,
        max_retries=0,
        backoff_base=0.01,
        dead_letter_path=dead_letter_path,
    )

    with caplog.at_level("ERROR", logger="undercurrent.sinks.webhook_sink"):
        sink.write_signal("req-fail", "ep-1", ProbeSignal())
        sink.write_signal("req-ok", "ep-1", ProbeSignal())
        start = time.monotonic()
        sink.close(timeout=2)
        elapsed = time.monotonic() - start

    assert elapsed < 2
    assert not sink._thread.is_alive()
    assert post_fn.delivered == ["req-ok"]  # the sender survived the failed dead-letter write
    assert any("could not write to dead-letter file" in record.message for record in caplog.records)


def test_close_returns_within_its_timeout_when_the_queue_is_full():
    gate = threading.Event()

    def _blocking_post(url, record):
        gate.wait(timeout=10)

    sink = WebhookLogSink("https://example.invalid/hook", post_fn=_blocking_post, queue_maxsize=1)
    sink.write_signal("req-1", "ep-1", ProbeSignal())  # occupies the worker
    time.sleep(0.05)
    sink.write_signal("req-1", "ep-1", ProbeSignal())  # fills the queue

    start = time.monotonic()
    sink.close(timeout=0.2)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    gate.set()
    sink.close(timeout=2)

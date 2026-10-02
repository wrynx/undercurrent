import copy
import json
import logging
import threading

import pytest

from tests.sinks._helpers import make_extraction_point, make_record
from undercurrent.core import ActivationRecord, Probe, ProbeAction, ProbeResult, ProbeSignal, RequestContext
from undercurrent.router import ProbeFactory, Router
from undercurrent.sinks import (
    DEFAULT_PROMPT_TEXT_KEYS,
    FileLogSink,
    LogSink,
    WebhookLogSink,
    chain,
    drop_keys,
    redact_keys,
    wire_router,
)
from undercurrent.spec import ExecutionMode

PROMPT = "my secret prompt: the launch code is 0000"


class RecordingPost:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, url, record):
        with self._lock:
            self.calls.append((url, record))
        if self.fail:
            raise ConnectionError("endpoint down")

    def payloads(self):
        return [json.dumps(record, default=str) for _, record in self.calls]


def _read_ndjson(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _prompt_signal():
    return ProbeSignal(
        action=ProbeAction.FLAG,
        metadata={"score": 0.9, "prompt": PROMPT, "nested": {"text": PROMPT, "prompt_len": 7}},
    )


def _prompt_result():
    return ProbeResult(
        request_id="req-1",
        extraction_point_name="ep-1",
        verdict={"flagged": True, "generated_text": PROMPT},
        signal_history=[_prompt_signal()],
        metadata={"prompt": PROMPT, "raw_scores": [0.1, 0.9]},
    )


# --- helpers ---------------------------------------------------------------


def test_redact_keys_replaces_values_at_any_depth_including_inside_lists():
    record = {
        "prompt": PROMPT,
        "payload": {"metadata": {"prompt": PROMPT, "prompt_len": 7}, "history": [{"prompt": PROMPT, "score": 1}]},
    }

    out = redact_keys("prompt")(record)

    assert out == {
        "prompt": "[REDACTED]",
        "payload": {
            "metadata": {"prompt": "[REDACTED]", "prompt_len": 7},
            "history": [{"prompt": "[REDACTED]", "score": 1}],
        },
    }


def test_redact_keys_custom_replacement_and_whole_subtree():
    out = redact_keys("metadata", replacement=None)({"payload": {"metadata": {"a": 1}, "action": "flag"}})
    assert out == {"payload": {"metadata": None, "action": "flag"}}


def test_drop_keys_dotted_path_only_matches_that_parent():
    record = {
        "payload": {
            "metadata": {"raw_scores": [1, 2], "score": 0.5},
            "verdict": {"raw_scores": [3]},
        }
    }

    out = drop_keys("metadata.raw_scores")(record)

    assert out == {"payload": {"metadata": {"score": 0.5}, "verdict": {"raw_scores": [3]}}}


def test_dotted_path_matches_through_lists():
    record = {"payload": {"signal_history": [{"metadata": {"prompt": PROMPT}}, {"metadata": {}}]}}
    out = drop_keys("signal_history.metadata.prompt")(record)
    assert out == {"payload": {"signal_history": [{"metadata": {}}, {"metadata": {}}]}}


def test_full_dotted_path_from_root():
    record = {"payload": {"metadata": {"prompt": PROMPT}}, "prompt": "keep"}
    assert redact_keys("payload.metadata.prompt")(record) == {
        "payload": {"metadata": {"prompt": "[REDACTED]"}},
        "prompt": "keep",
    }


def test_helpers_do_not_mutate_their_input():
    record = {"payload": {"metadata": {"prompt": PROMPT, "raw_scores": [1]}}}
    snapshot = copy.deepcopy(record)

    redact_keys("prompt")(record)
    drop_keys("raw_scores")(record)

    assert record == snapshot


def test_chain_composes_left_to_right_and_short_circuits_on_none():
    seen = []

    def spy(record):
        seen.append(record)
        return record

    fn = chain(drop_keys("raw_scores"), redact_keys("prompt"), spy)
    assert fn({"prompt": PROMPT, "raw_scores": [1]}) == {"prompt": "[REDACTED]"}
    assert seen == [{"prompt": "[REDACTED]"}]

    seen.clear()
    assert chain(lambda r: None, spy)({"a": 1}) is None
    assert seen == []


@pytest.mark.parametrize("bad", [(), ("",), ("a..b",), (".a",)])
def test_invalid_key_patterns_are_rejected(bad):
    with pytest.raises(ValueError):
        redact_keys(*bad)


def test_default_prompt_text_keys_cover_the_adapter_prompt_key():
    assert "prompt" in DEFAULT_PROMPT_TEXT_KEYS
    assert "prompt_len" not in DEFAULT_PROMPT_TEXT_KEYS


# --- FileLogSink -----------------------------------------------------------


def test_file_sink_default_keeps_prompt_text(tmp_path):
    path = tmp_path / "log.ndjson"
    FileLogSink(path).write_signal("req-1", "ep-1", _prompt_signal())
    assert PROMPT in path.read_text(encoding="utf-8")


def test_file_sink_applies_redact_keys(tmp_path):
    path = tmp_path / "log.ndjson"
    sink = FileLogSink(path, redact=redact_keys("prompt", "text"))

    sink.write_signal("req-1", "ep-1", _prompt_signal())

    [record] = _read_ndjson(path)
    assert record["payload"]["metadata"]["prompt"] == "[REDACTED]"
    assert record["payload"]["metadata"]["nested"] == {"text": "[REDACTED]", "prompt_len": 7}
    assert record["payload"]["metadata"]["score"] == 0.9


def test_file_sink_redact_returning_none_drops_record(tmp_path):
    path = tmp_path / "log.ndjson"
    sink = FileLogSink(path, redact=lambda r: None if r["extraction_point_name"] == "debug" else r)

    sink.write_signal("req-1", "debug", ProbeSignal())
    sink.write_signal("req-1", "ep-1", ProbeSignal())

    records = _read_ndjson(path)
    assert [r["extraction_point_name"] for r in records] == ["ep-1"]
    assert sink.redaction_error_count == 0


def test_file_sink_redact_exception_fails_closed_and_is_counted(tmp_path, caplog):
    path = tmp_path / "log.ndjson"

    def boom(record):
        raise RuntimeError(f"cannot handle {record['payload']['metadata']['prompt']}")

    sink = FileLogSink(path, redact=boom)
    with caplog.at_level(logging.WARNING, logger="undercurrent.sinks"):
        sink.write_signal("req-1", "ep-1", _prompt_signal())
        sink.write_signal("req-1", "ep-1", _prompt_signal())

    assert _read_ndjson(path) == []
    assert sink.redaction_error_count == 2
    assert "RuntimeError" in caplog.text
    assert PROMPT not in caplog.text


def test_redact_returning_non_dict_fails_closed(tmp_path):
    path = tmp_path / "log.ndjson"
    sink = FileLogSink(path, redact=lambda r: json.dumps(r))
    sink.write_signal("req-1", "ep-1", _prompt_signal())
    assert _read_ndjson(path) == []
    assert sink.redaction_error_count == 1


def test_redact_mutating_its_argument_does_not_touch_callers_objects(tmp_path):
    path = tmp_path / "log.ndjson"

    def mutate(record):
        record["payload"]["metadata"].clear()
        record["mutated"] = True
        return record

    sink = FileLogSink(path, redact=mutate)
    signal = _prompt_signal()
    signal_snapshot = copy.deepcopy(signal)
    raw = {"kind": "signal", "payload": {"metadata": {"prompt": PROMPT}}}
    raw_snapshot = copy.deepcopy(raw)

    sink.write_signal("req-1", "ep-1", signal)
    sink.write_raw(raw)

    assert signal == signal_snapshot
    assert raw == raw_snapshot
    assert all(r["mutated"] and r["payload"]["metadata"] == {} for r in _read_ndjson(path))


def test_custom_sink_can_reuse_apply_redaction():
    class ListSink(LogSink):
        def __init__(self, redact=None):
            super().__init__(redact=redact)
            self.records = []

        def write_signal(self, request_id, extraction_point_name, signal):
            record = self._apply_redaction({"payload": {"metadata": dict(signal.metadata)}})
            if record is not None:
                self.records.append(record)

        def write_result(self, request_id, extraction_point_name, result):
            pass

    sink = ListSink(redact=redact_keys("prompt"))
    sink.write_signal("req-1", "ep-1", _prompt_signal())
    assert sink.records[0]["payload"]["metadata"]["prompt"] == "[REDACTED]"

    class NoInitSink(LogSink):
        def __init__(self):
            pass

        write_signal = write_result = lambda self, *a: None

    bare = NoInitSink()
    assert bare._apply_redaction({"a": 1}) == {"a": 1}
    assert bare.redaction_error_count == 0


# --- WebhookLogSink --------------------------------------------------------


def test_webhook_default_redacts_prompt_text_from_every_payload():
    post = RecordingPost()
    sink = WebhookLogSink("https://example.com/hook", post_fn=post)

    sink.write_signal("req-1", "ep-1", _prompt_signal())
    sink.write_result("req-1", "ep-1", _prompt_result())
    sink.close(timeout=5)

    payloads = post.payloads()
    assert len(payloads) == 2
    assert all(PROMPT not in p for p in payloads)
    signal_record = post.calls[0][1]
    assert signal_record["payload"]["metadata"]["prompt"] == "[REDACTED]"
    assert signal_record["payload"]["metadata"]["nested"]["prompt_len"] == 7
    result_record = post.calls[1][1]
    assert result_record["payload"]["verdict"] == {"flagged": True, "generated_text": "[REDACTED]"}
    assert result_record["payload"]["metadata"]["raw_scores"] == [0.1, 0.9]


def test_webhook_include_prompt_text_sends_it():
    post = RecordingPost()
    sink = WebhookLogSink("https://example.com/hook", post_fn=post, include_prompt_text=True)

    sink.write_signal("req-1", "ep-1", _prompt_signal())
    sink.close(timeout=5)

    [payload] = post.payloads()
    assert PROMPT in payload


def test_webhook_custom_redact_runs_after_default():
    post = RecordingPost()
    seen = []

    def custom(record):
        seen.append(json.dumps(record))
        return drop_keys("metadata.raw_scores")(record)

    sink = WebhookLogSink("https://example.com/hook", post_fn=post, redact=custom)
    sink.write_result("req-1", "ep-1", _prompt_result())
    sink.close(timeout=5)

    assert PROMPT not in seen[0]
    [(_, record)] = post.calls
    assert "raw_scores" not in record["payload"]["metadata"]
    assert PROMPT not in post.payloads()[0]


def test_webhook_redact_none_drops_without_sending():
    post = RecordingPost()
    sink = WebhookLogSink("https://example.com/hook", post_fn=post, redact=lambda r: None)
    sink.write_signal("req-1", "ep-1", ProbeSignal())
    sink.close(timeout=5)
    assert post.calls == []
    assert sink.redaction_error_count == 0


def test_webhook_redact_exception_fails_closed_never_sends_or_dead_letters(tmp_path):
    post = RecordingPost()
    dead_letter = tmp_path / "dead.ndjson"

    def boom(record):
        raise ValueError("bad")

    sink = WebhookLogSink(
        "https://example.com/hook",
        post_fn=post,
        redact=boom,
        include_prompt_text=True,
        dead_letter_path=dead_letter,
    )
    sink.write_signal("req-1", "ep-1", _prompt_signal())
    sink.write_result("req-1", "ep-1", _prompt_result())
    sink.close(timeout=5)

    assert post.calls == []
    assert _read_ndjson(dead_letter) == []
    assert sink.redaction_error_count == 2
    assert sink.dropped_count == 0


def test_webhook_dead_letter_file_is_redacted(tmp_path):
    post = RecordingPost(fail=True)
    dead_letter = tmp_path / "dead.ndjson"
    sink = WebhookLogSink(
        "https://example.com/hook",
        post_fn=post,
        max_retries=1,
        backoff_base=0.0,
        dead_letter_path=dead_letter,
        redact=drop_keys("raw_scores"),
    )

    sink.write_result("req-1", "ep-1", _prompt_result())
    sink.close(timeout=5)

    assert len(post.calls) == 2
    assert PROMPT not in dead_letter.read_text(encoding="utf-8")
    [record] = _read_ndjson(dead_letter)
    assert record["payload"]["metadata"] == {"prompt": "[REDACTED]"}


def test_webhook_without_dead_letter_logs_only_redacted_record(caplog):
    post = RecordingPost(fail=True)
    sink = WebhookLogSink("https://example.com/hook", post_fn=post, max_retries=0)
    with caplog.at_level(logging.ERROR, logger="undercurrent.sinks"):
        sink.write_signal("req-1", "ep-1", _prompt_signal())
        sink.close(timeout=5)
    assert "permanent delivery failure" in caplog.text
    assert PROMPT not in caplog.text


def test_webhook_logs_policy_at_info_without_leaking_url_secrets(caplog):
    with caplog.at_level(logging.INFO, logger="undercurrent.sinks"):
        WebhookLogSink("https://user:pw@example.com/hook?token=s3cret", post_fn=RecordingPost()).close(timeout=5)
        WebhookLogSink("https://example.com/hook", post_fn=RecordingPost(), include_prompt_text=True).close(timeout=5)

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(messages) == 2
    assert "redacted" in messages[0] and "include_prompt_text=True" in messages[0]
    assert "https://example.com" in messages[0]
    assert "s3cret" not in messages[0] and "pw" not in messages[0]
    assert "INCLUDED" in messages[1]


# --- router integration ----------------------------------------------------


class PromptEchoProbe(Probe):
    """A careless probe: copies the adapter's prompt into everything it emits."""

    probe_kind = "trajectory"

    def __init__(self) -> None:
        super().__init__()
        self._prompt = None
        self._count = 0

    def on_start(self, request_ctx: RequestContext) -> None:
        self._prompt = request_ctx.prompt_metadata["prompt"]

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        self._count += 1
        return ProbeSignal(metadata={"prompt": self._prompt, "token_pos": record.token_pos})

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"count": self._count, "text": self._prompt},
            metadata=dict(request_ctx.prompt_metadata),
        )


def _run_prompt_echo_request(sink):
    router = Router({"echo": ProbeFactory(PromptEchoProbe, {})})
    try:
        wire_router(router, sink)
        point = make_extraction_point(name="ep-1", probe_type="echo", execution_mode=ExecutionMode.ASYNC)
        ctx = RequestContext(
            request_id="req-1",
            prompt_metadata={"model": "gpt2", "prompt": PROMPT, "prompt_len": 9},
            extraction_point_config=None,
        )
        router.register_request("req-1", [point], ctx)
        for token_pos in range(3):
            router.route(make_record(point, token_pos=token_pos))
        router.end_request("req-1")
    finally:
        router.shutdown()
    sink.close(timeout=5)


def test_router_end_to_end_webhook_default_never_sends_prompt():
    post = RecordingPost()
    _run_prompt_echo_request(WebhookLogSink("https://example.com/hook", post_fn=post))

    kinds = [record["kind"] for _, record in post.calls]
    assert kinds.count("signal") == 3 and kinds.count("result") == 1
    assert all(PROMPT not in payload for payload in post.payloads())
    result = next(record for _, record in post.calls if record["kind"] == "result")
    assert result["payload"]["metadata"] == {"model": "gpt2", "prompt": "[REDACTED]", "prompt_len": 9}
    assert result["payload"]["verdict"] == {"count": 3, "text": "[REDACTED]"}


def test_router_end_to_end_webhook_opt_in_sends_prompt():
    post = RecordingPost()
    _run_prompt_echo_request(WebhookLogSink("https://example.com/hook", post_fn=post, include_prompt_text=True))
    assert len(post.calls) == 4
    assert all(PROMPT in payload for payload in post.payloads())

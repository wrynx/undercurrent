import json

from undercurrent.core import ProbeAction, ProbeResult, ProbeSignal
from undercurrent.sinks import FileLogSink


class _FakeTensor:
    """Duck-typed tensor stand-in (shape + dtype), since torch/numpy aren't
    a dependency of this package or test environment."""

    def __init__(self, shape, dtype="float32"):
        self.shape = shape
        self.dtype = dtype


def _read_ndjson(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_write_signal_produces_one_parseable_ndjson_line_with_correct_fields(tmp_path):
    path = tmp_path / "log.ndjson"
    sink = FileLogSink(path)
    signal = ProbeSignal(action=ProbeAction.FLAG, metadata={"score": 0.9}, confidence=0.9)

    sink.write_signal("req-1", "ep-1", signal)

    records = _read_ndjson(path)
    assert len(records) == 1
    record = records[0]
    assert record["kind"] == "signal"
    assert record["request_id"] == "req-1"
    assert record["extraction_point_name"] == "ep-1"
    assert isinstance(record["timestamp"], (int, float))
    assert record["payload"]["action"] == "flag"
    assert record["payload"]["metadata"] == {"score": 0.9}
    assert record["payload"]["confidence"] == 0.9


def test_write_result_produces_one_parseable_ndjson_line_with_correct_fields(tmp_path):
    path = tmp_path / "log.ndjson"
    sink = FileLogSink(path)
    result = ProbeResult(
        request_id="req-1",
        extraction_point_name="ep-1",
        verdict={"final_mean": 0.42, "count": 3},
        signal_history=[ProbeSignal()],
        metadata={"threshold": 0.8},
    )

    sink.write_result("req-1", "ep-1", result)

    records = _read_ndjson(path)
    assert len(records) == 1
    record = records[0]
    assert record["kind"] == "result"
    assert record["request_id"] == "req-1"
    assert record["extraction_point_name"] == "ep-1"
    assert record["payload"]["verdict"] == {"final_mean": 0.42, "count": 3}
    assert record["payload"]["metadata"] == {"threshold": 0.8}
    assert len(record["payload"]["signal_history"]) == 1


def test_multiple_writes_append_one_line_each_in_order(tmp_path):
    path = tmp_path / "log.ndjson"
    sink = FileLogSink(path)

    for i in range(5):
        sink.write_signal("req-1", "ep-1", ProbeSignal(metadata={"i": i}))

    records = _read_ndjson(path)
    assert [r["payload"]["metadata"]["i"] for r in records] == [0, 1, 2, 3, 4]


def test_tensor_like_field_is_summarized_not_dumped_raw(tmp_path):
    path = tmp_path / "log.ndjson"
    sink = FileLogSink(path)
    tensor = _FakeTensor(shape=(4, 8), dtype="float16")
    signal = ProbeSignal(metadata={"activation": tensor})

    sink.write_signal("req-1", "ep-1", signal)

    record = _read_ndjson(path)[0]
    summary = record["payload"]["metadata"]["activation"]
    assert summary == {"__summary__": "tensor", "type": "_FakeTensor", "shape": [4, 8], "dtype": "float16"}


def test_concurrent_writes_from_multiple_threads_never_interleave_a_line(tmp_path):
    import threading

    path = tmp_path / "log.ndjson"
    sink = FileLogSink(path)

    def _writer(n):
        for i in range(20):
            sink.write_signal("req-1", "ep-1", ProbeSignal(metadata={"writer": n, "i": i}))

    threads = [threading.Thread(target=_writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = _read_ndjson(path)  # json.loads raising would mean a line got interleaved/corrupted
    assert len(records) == 8 * 20

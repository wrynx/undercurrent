"""VLLMEngineAdapter.shutdown() closes the engine, not just the event loop.

Leaving vLLM's AsyncLLM to garbage collection keeps its EngineCore subprocess
alive and can block forever in ZeroMQ's Context.term() (seen on GPU). These
tests use a fake engine, so they run without vLLM or a GPU.
"""

import asyncio
import threading

import pytest

from undercurrent.adapters.vllm import VLLMEngineAdapter


class FakeEngine:
    def __init__(self, raise_on_shutdown: bool = False):
        self.shutdown_calls: list[float | None] = []
        self.raise_on_shutdown = raise_on_shutdown

    def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_calls.append(timeout)
        if self.raise_on_shutdown:
            raise RuntimeError("engine already dead")


class FakeEngineNoTimeout:
    def __init__(self):
        self.calls = 0

    def shutdown(self) -> None:
        self.calls += 1


def _started_adapter(engine) -> VLLMEngineAdapter:
    adapter = VLLMEngineAdapter()
    adapter._engine = engine
    adapter._loop = asyncio.new_event_loop()
    adapter._loop_thread = threading.Thread(target=adapter._loop.run_forever, daemon=True)
    adapter._loop_thread.start()
    return adapter


@pytest.fixture(autouse=True)
def _allow_any_vllm(monkeypatch):
    # The constructor checks the installed vLLM version; these tests don't need vLLM at all.
    monkeypatch.setenv("UNDERCURRENT_ALLOW_UNSUPPORTED_VLLM", "1")


def test_shutdown_closes_engine_then_loop():
    engine = FakeEngine()
    adapter = _started_adapter(engine)
    loop, thread = adapter._loop, adapter._loop_thread

    adapter.shutdown(timeout=5.0)

    assert engine.shutdown_calls == [5.0]
    assert not thread.is_alive()
    assert loop.is_closed()
    assert adapter._engine is None and adapter._loop is None and adapter._loop_thread is None


def test_shutdown_is_idempotent():
    engine = FakeEngine()
    adapter = _started_adapter(engine)
    adapter.shutdown()
    adapter.shutdown()
    assert len(engine.shutdown_calls) == 1


def test_shutdown_still_stops_the_loop_when_engine_shutdown_fails(caplog):
    adapter = _started_adapter(FakeEngine(raise_on_shutdown=True))
    loop, thread = adapter._loop, adapter._loop_thread

    adapter.shutdown()

    assert not thread.is_alive()
    assert loop.is_closed()
    assert "engine shutdown failed" in caplog.text


def test_shutdown_supports_engines_without_a_timeout_parameter():
    engine = FakeEngineNoTimeout()
    adapter = _started_adapter(engine)
    adapter.shutdown()
    assert engine.calls == 1


def test_shutdown_before_load_model_is_a_no_op():
    VLLMEngineAdapter().shutdown()

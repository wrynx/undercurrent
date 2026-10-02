"""Router context managers: `Router.request(...)`, `Router.on_request_end`,
`with Router(...)`, and use-after-shutdown errors."""

import logging
import threading

import pytest

from tests.router._helpers import make_extraction_point
from undercurrent.core import Probe, ProbeResult
from undercurrent.router import ProbeFactory, RequestHandle, Router, RouterError
from undercurrent.spec import ProbeSpec


class _Boom(Exception):
    pass


class _RaisingOnEndProbe(Probe):
    probe_kind = "trajectory"

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        return None

    def on_end(self, request_ctx):
        raise RuntimeError("on_end failed")


@pytest.fixture
def router(probe_registry):
    r = Router(probe_registry)
    yield r
    r.shutdown(wait=True)


def test_results_available_after_block(router, make_record):
    point = make_extraction_point(name="ep-1")
    with router.request([point], request_id="req-1") as req:
        assert isinstance(req, RequestHandle)
        req.route(make_record(request_id="req-1", extraction_point_name="ep-1"))
        with pytest.raises(RouterError, match="only available after"):
            _ = req.results

    assert set(req.results) == {"ep-1"}
    assert isinstance(req.results["ep-1"], ProbeResult)
    assert req.results["ep-1"].request_id == "req-1"
    # end_request already ran: the request is torn down.
    with pytest.raises(RouterError):
        router.get_probe("req-1", "ep-1")


def test_prompt_metadata_reaches_request_ctx(router):
    with router.request([make_extraction_point()], prompt_metadata={"prompt": "hi"}) as req:
        assert req.request_ctx.prompt_metadata == {"prompt": "hi"}
        assert req.request_ctx.request_id == req.request_id


def test_end_request_called_when_body_raises(router):
    seen = []
    router.on_request_end(lambda request_id, results: seen.append(request_id))

    with pytest.raises(_Boom), router.request([make_extraction_point()], request_id="req-1") as req:
        raise _Boom()

    assert seen == ["req-1"]
    assert set(req.results) == {"ep-1"}
    with pytest.raises(RouterError):
        router.get_probe("req-1", "ep-1")


def test_end_request_failure_does_not_mask_body_exception(make_record, caplog):
    router = Router({"raising": ProbeFactory(_RaisingOnEndProbe)})
    point = make_extraction_point(probe_type="raising")
    with (
        caplog.at_level(logging.ERROR, logger="undercurrent.router.request"),
        pytest.raises(_Boom),
        router.request([point], request_id="req-1") as req,
    ):
        raise _Boom()
    assert "end_request('req-1') raised" in caplog.text
    with pytest.raises(RouterError, match="end_request failed"):
        _ = req.results
    router.shutdown()


def test_end_request_failure_propagates_without_body_exception():
    router = Router({"raising": ProbeFactory(_RaisingOnEndProbe)})
    point = make_extraction_point(probe_type="raising")
    with pytest.raises(RuntimeError, match="on_end failed"), router.request([point]):
        pass
    router.shutdown()


def test_auto_request_id(router):
    with router.request([make_extraction_point()]) as req_a:
        pass
    with router.request([make_extraction_point()]) as req_b:
        pass
    assert len(req_a.request_id) == 32
    int(req_a.request_id, 16)  # uuid4 hex
    assert req_a.request_id != req_b.request_id


def test_route_rejects_mismatched_request_id(router, make_record):
    with (
        router.request([make_extraction_point()], request_id="req-1") as req,
        pytest.raises(RouterError, match="does not match"),
    ):
        req.route(make_record(request_id="req-other"))


def test_route_outside_block_raises(router, make_record):
    with router.request([make_extraction_point()], request_id="req-1") as req:
        pass
    with pytest.raises(RouterError, match="not active"):
        req.route(make_record(request_id="req-1"))


def test_handle_is_single_use(router):
    handle = router.request([make_extraction_point()])
    with handle:
        pass
    with pytest.raises(RouterError, match="only be entered once"), handle:
        pass


def test_probe_spec_accepted(router, make_request_ctx):
    spec = ProbeSpec(
        version="1",
        extraction_points=(make_extraction_point(name="ep-a"), make_extraction_point(name="ep-b")),
    )
    with router.request(spec) as req:
        pass
    assert set(req.results) == {"ep-a", "ep-b"}

    router.register_request("req-1", spec, make_request_ctx())
    assert set(router.end_request("req-1")) == {"ep-a", "ep-b"}
    # A one-shot iterator works too: register_request converts to a list.
    router.register_request("req-2", iter(spec), make_request_ctx(request_id="req-2"))
    assert set(router.end_request("req-2")) == {"ep-a", "ep-b"}


def test_listener_called_once_with_results(router, make_request_ctx):
    calls = []
    router.on_request_end(lambda request_id, results: calls.append((request_id, results)))

    router.register_request("req-1", [make_extraction_point()], make_request_ctx())
    results = router.end_request("req-1")

    assert len(calls) == 1
    request_id, seen = calls[0]
    assert request_id == "req-1"
    assert seen == results
    assert seen["ep-1"] is results["ep-1"]


def test_listeners_run_in_registration_order(router):
    order = []
    router.on_request_end(lambda request_id, results: order.append("a"))
    router.on_request_end(lambda request_id, results: order.append("b"))
    with router.request([make_extraction_point()]):
        pass
    assert order == ["a", "b"]


def test_listener_exception_is_isolated(router, caplog):
    seen = []

    def bad(request_id, results):
        raise RuntimeError("listener failed")

    router.on_request_end(bad)
    router.on_request_end(lambda request_id, results: seen.append(request_id))

    with (
        caplog.at_level(logging.ERROR, logger="undercurrent.router.router"),
        router.request([make_extraction_point()], request_id="req-1") as req,
    ):
        pass

    assert seen == ["req-1"]
    assert set(req.results) == {"ep-1"}
    assert "listener failed" in caplog.text


def test_unsubscribe(router):
    calls = []

    def listener(request_id, results):
        calls.append(request_id)

    remove = router.on_request_end(listener)
    with router.request([make_extraction_point()], request_id="req-1"):
        pass
    remove()
    remove()  # idempotent
    with router.request([make_extraction_point()], request_id="req-2"):
        pass
    assert calls == ["req-1"]


def test_unsubscribe_removes_only_its_own_registration(router):
    calls = []

    def listener(request_id, results):
        calls.append(request_id)

    remove_first = router.on_request_end(listener)
    router.on_request_end(listener)
    remove_first()
    with router.request([make_extraction_point()], request_id="req-1"):
        pass
    assert calls == ["req-1"]


def test_listener_runs_outside_router_lock(router, make_request_ctx):
    """A listener that calls back into the router -- including from another
    thread, which an RLock would not let through -- must not deadlock."""
    outcome = {}

    def listener(request_id, results):
        if request_id != "req-1":
            return

        def other_thread():
            outcome["acquired"] = router._lock.acquire(timeout=2)
            if outcome["acquired"]:
                router._lock.release()
            router.register_request("req-follow-up", [make_extraction_point()], make_request_ctx("req-follow-up"))
            router.end_request("req-follow-up")
            outcome["done"] = True

        t = threading.Thread(target=other_thread)
        t.start()
        t.join(timeout=5)

    remove = router.on_request_end(listener)
    router.register_request("req-1", [make_extraction_point()], make_request_ctx())
    router.end_request("req-1")
    remove()

    assert outcome == {"acquired": True, "done": True}


def test_router_with_block_shuts_down(probe_registry, make_request_ctx):
    with Router(probe_registry) as router:
        router.register_request("req-1", [make_extraction_point()], make_request_ctx())
    assert router._closed
    assert router._executor._shutdown


def test_use_after_shutdown_raises(probe_registry, make_request_ctx, make_record):
    router = Router(probe_registry)
    router.register_request("req-1", [make_extraction_point()], make_request_ctx())
    router.shutdown()

    msg = "router has been shut down; create a new Router"
    with pytest.raises(RouterError, match=msg):
        router.register_request("req-2", [make_extraction_point()], make_request_ctx("req-2"))
    with pytest.raises(RouterError, match=msg):
        router.route(make_record(request_id="req-1"))
    with pytest.raises(RouterError, match=msg):
        router.end_request("req-1")
    with pytest.raises(RouterError, match=msg), router.request([make_extraction_point()]):
        pass

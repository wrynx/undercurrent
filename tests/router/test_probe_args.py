"""An extraction point's `probe_args` are merged over the factory's kwargs
when the router spawns its probe (the point wins)."""

from dataclasses import replace

from tests.router._helpers import make_extraction_point
from undercurrent.core import Probe, ProbeResult
from undercurrent.router import ProbeFactory, Router


class _KwargsProbe(Probe):
    probe_kind = "trajectory"

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        return None

    def on_end(self, request_ctx):
        return ProbeResult(request_id=self.request_id, extraction_point_name=self.extraction_point_name, verdict=None)


def test_point_probe_args_override_factory_kwargs(make_request_ctx):
    router = Router({"kw": ProbeFactory(_KwargsProbe, {"threshold": 0.1, "seed": 3})})
    point = replace(make_extraction_point(name="ep-1", probe_type="kw"), probe_args={"threshold": 0.9, "extra": "x"})

    router.register_request("req-1", [point], make_request_ctx())

    assert router.get_probe("req-1", "ep-1").kwargs == {"threshold": 0.9, "seed": 3, "extra": "x"}


def test_empty_probe_args_uses_factory_kwargs(make_request_ctx):
    router = Router({"kw": ProbeFactory(_KwargsProbe, {"threshold": 0.1})})
    router.register_request("req-1", [make_extraction_point(name="ep-1", probe_type="kw")], make_request_ctx())

    assert router.get_probe("req-1", "ep-1").kwargs == {"threshold": 0.1}


def test_factory_spawn_merges_extra_kwargs():
    factory = ProbeFactory(_KwargsProbe, {"a": 1, "b": 2})
    probe = factory.spawn("req-1", "ep-1", b=20, c=30)
    assert probe.kwargs == {"a": 1, "b": 20, "c": 30}
    assert factory.probe_kwargs == {"a": 1, "b": 2}

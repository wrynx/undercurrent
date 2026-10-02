"""@probe: plain functions as single_shot probes."""

from dataclasses import replace

import pytest

from tests.router._helpers import make_extraction_point
from undercurrent.core import (
    ActivationRecord,
    Probe,
    ProbeAction,
    ProbeFactory,
    ProbeRegistry,
    ProbeResult,
    ProbeSignal,
    RequestContext,
    default_registry,
    probe,
    register_probe,
)
from undercurrent.core.function_probe import FunctionProbe
from undercurrent.router import Router, RouterError
from undercurrent.spec import ExecutionMode, ProbeKind


@pytest.fixture
def registry():
    return ProbeRegistry(load_entry_points=False)


@pytest.fixture
def ctx():
    return RequestContext(request_id="req-1", prompt_metadata={}, extraction_point_config=None)


def _record(value=0.0, token_pos=0, request_id="req-1", extraction_point_name="ep-1"):
    return ActivationRecord(
        request_id=request_id,
        extraction_point_name=extraction_point_name,
        layer=5,
        token_pos=token_pos,
        tensor_type="residual_stream",
        tensor=[value],
        is_generated=True,
    )


def _score(record) -> float:
    return record.tensor[0]


def _run(cls, values, ctx, **kwargs):
    instance = cls.spawn("req-1", "ep-1", **kwargs)
    instance.on_start(ctx)
    signals = [instance.on_activation(_record(v, i)) for i, v in enumerate(values)]
    return signals, instance.on_end(ctx)


# -- the generated class -----------------------------------------------------


def test_decorator_returns_single_shot_probe_subclass(registry):
    @probe("tox", threshold=0.5, registry=registry)
    def toxicity(record) -> float:
        """Toxicity score."""
        return record.tensor[0]

    assert isinstance(toxicity, type) and issubclass(toxicity, Probe)
    assert issubclass(toxicity, FunctionProbe)
    assert toxicity.probe_kind == "single_shot"
    assert toxicity.__name__ == "toxicity"
    assert toxicity.__doc__ == "Toxicity score."
    assert toxicity.__module__ == __name__
    assert toxicity.fn(_record(0.25)) == 0.25
    assert registry.get("tox").probe_cls is toxicity


def test_bare_decorator_uses_function_name(registry, monkeypatch):
    monkeypatch.setattr("undercurrent.core.function_probe.default_registry", registry)

    @probe
    def bare_probe(record):
        return None

    assert registry.get("bare_probe").probe_cls is bare_probe


def test_name_defaults_to_function_name(registry):
    @probe(registry=registry)
    def my_scorer(record):
        return 0.0

    assert "my_scorer" in registry


def test_registers_in_default_registry_by_default():
    @probe("fn_probe_default_registry_test")
    def scorer(record):
        return 0.0

    try:
        assert default_registry.get("fn_probe_default_registry_test").probe_cls is scorer
    finally:
        default_registry.unregister("fn_probe_default_registry_test")


def test_register_false_skips_registration(registry, monkeypatch):
    monkeypatch.setattr("undercurrent.core.function_probe.default_registry", registry)

    @probe("unregistered", register=False)
    def scorer(record):
        return 0.0

    assert "unregistered" not in registry
    assert "unregistered" not in default_registry
    # ... and the class composes with register_probe.
    register_probe("composed", override=True)(scorer)
    try:
        assert default_registry.get("composed").probe_cls is scorer
    finally:
        default_registry.unregister("composed")


def test_custom_registry_does_not_touch_default(registry):
    @probe("only_in_custom", registry=registry)
    def scorer(record):
        return 0.0

    assert "only_in_custom" in registry
    assert "only_in_custom" not in default_registry


def test_redecorating_same_function_is_a_redefinition(registry):
    for _ in range(2):  # e.g. a re-run notebook cell

        @probe("rerun", registry=registry)
        def scorer(record):
            return 0.0

    assert registry.get("rerun").probe_cls is scorer


# -- return-value mapping ----------------------------------------------------


def test_none_emits_nothing_and_is_not_counted(ctx):
    @probe(register=False, threshold=0.5)
    def silent(record):
        return None

    signals, result = _run(silent, [0.9, 0.9], ctx)
    assert signals == [None, None]
    assert result.signal_history == []
    assert result.verdict == {"flagged": False, "max_score": None, "n": 0}


def test_bool_true_flags_and_false_continues(ctx):
    @probe("boolish", register=False)
    def boolish(record):
        return record.tensor[0] > 0

    signals, result = _run(boolish, [1.0, -1.0], ctx)
    assert signals[0].action is ProbeAction.ABORT
    assert signals[0].confidence == 1.0
    assert signals[0].metadata == {"score": None, "probe": "boolish"}
    assert signals[1].action is ProbeAction.CONTINUE
    assert signals[1].confidence is None
    assert result.verdict == {"flagged": True, "max_score": None, "n": 2}


def test_float_at_threshold_flags(ctx):
    cls = probe("t", threshold=0.5, register=False)(_score)
    signals, result = _run(cls, [0.49, 0.5, 0.7], ctx)
    assert [s.action for s in signals] == [ProbeAction.CONTINUE, ProbeAction.ABORT, ProbeAction.ABORT]
    assert [s.confidence for s in signals] == [0.49, 0.5, 0.7]
    assert signals[1].metadata == {"score": 0.5, "probe": "t"}
    assert result.verdict == {"flagged": True, "max_score": 0.7, "n": 3}


def test_int_is_a_score(ctx):
    @probe(register=False, threshold=2)
    def count(record):
        return int(record.tensor[0])

    signals, _ = _run(count, [1.0, 2.0], ctx)
    assert [s.action for s in signals] == [ProbeAction.CONTINUE, ProbeAction.ABORT]
    assert signals[1].metadata["score"] == 2.0


def test_float_without_threshold_is_recorded_never_flagged(ctx):
    cls = probe("nt", register=False)(_score)
    signals, result = _run(cls, [0.1, 100.0], ctx)
    assert all(s.action is ProbeAction.CONTINUE for s in signals)
    assert [s.metadata["score"] for s in result.signal_history] == [0.1, 100.0]
    assert result.verdict == {"flagged": False, "max_score": 100.0, "n": 2}


def test_action_flag(ctx):
    cls = probe("f", threshold=0.5, action="flag", register=False)(_score)
    signals, result = _run(cls, [0.9], ctx)
    assert signals[0].action is ProbeAction.FLAG
    assert result.verdict["flagged"] is True


def test_probe_signal_passes_through_untouched(ctx):
    custom = ProbeSignal(action=ProbeAction.FLAG, metadata={"why": "custom"}, confidence=0.3)

    @probe(register=False, threshold=0.0)
    def full_control(record):
        return custom

    signals, result = _run(full_control, [1.0], ctx)
    assert signals[0] is custom
    assert result.signal_history == [custom]
    assert result.verdict == {"flagged": True, "max_score": None, "n": 1}


def test_bad_return_type_raises_type_error(ctx):
    @probe(register=False)
    def returns_list(record):
        return [1.0]

    instance = returns_list.spawn("req-1", "ep-1")
    with pytest.raises(TypeError, match=r"returns_list returned list.*float, int, bool, ProbeSignal or None"):
        instance.on_activation(_record())


def test_function_exceptions_propagate(ctx):
    @probe(register=False)
    def boom(record):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        boom.spawn("req-1", "ep-1").on_activation(_record())


def test_verdict_and_result_shape(ctx):
    cls = probe("shape", threshold=0.5, register=False)(_score)
    _, result = _run(cls, [0.2, 0.4], ctx)
    assert isinstance(result, ProbeResult)
    assert result.request_id == "req-1"
    assert result.extraction_point_name == "ep-1"
    assert result.verdict == {"flagged": False, "max_score": 0.4, "n": 2}
    assert len(result.signal_history) == 2


def test_on_end_without_activations(ctx):
    cls = probe("empty", register=False)(_score)
    instance = cls.spawn("req-1", "ep-1")
    instance.on_start(ctx)
    assert instance.on_end(ctx).verdict == {"flagged": False, "max_score": None, "n": 0}


def test_instances_are_isolated(ctx):
    cls = probe("iso", threshold=0.5, register=False)(_score)
    a = cls.spawn("req-1", "ep-1")
    b = cls.spawn("req-2", "ep-1")
    a.on_activation(_record(0.9))
    assert b.on_end(RequestContext("req-2", {}, None)).verdict == {"flagged": False, "max_score": None, "n": 0}


# -- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda: 0.0,
        lambda record, other: 0.0,
        lambda record, other=1: 0.0,
        lambda *records: 0.0,
        lambda record, *rest: 0.0,
    ],
    ids=["no-params", "two-positional", "positional-default", "var-positional-only", "var-positional"],
)
def test_bad_signature_rejected(fn):
    with pytest.raises(TypeError, match=r"@probe: .*(exactly one positional parameter|positional argument)"):
        probe(register=False)(fn)


def test_reserved_parameter_names_rejected():
    with pytest.raises(TypeError, match="threshold"):

        @probe(register=False)
        def scorer(record, *, threshold=0.5):
            return 0.0


def test_decorating_a_class_rejected():
    with pytest.raises(TypeError, match="decorates a function"):
        probe(register=False)(Probe)


@pytest.mark.parametrize("threshold", ["0.5", True, [0.5]])
def test_bad_threshold_rejected(threshold):
    with pytest.raises(TypeError, match="threshold must be a number"):
        probe(threshold=threshold)


def test_bad_action_rejected():
    with pytest.raises(ValueError, match=r"action must be one of \['abort', 'flag'\]"):
        probe(action="continue")


def test_bad_name_rejected():
    with pytest.raises(ValueError, match="non-empty string"):
        probe("")


# -- spawn kwargs ------------------------------------------------------------


def test_spawn_kwargs_override_threshold_and_action(ctx):
    cls = probe("o", threshold=0.9, register=False)(_score)
    signals, _ = _run(cls, [0.6], ctx, threshold=0.5, action="flag")
    assert signals[0].action is ProbeAction.FLAG
    signals, _ = _run(cls, [0.6], ctx, threshold=None)
    assert signals[0].action is ProbeAction.CONTINUE


def test_spawn_kwargs_validated():
    cls = probe("v", register=False)(_score)
    with pytest.raises(TypeError, match="threshold must be a number"):
        cls.spawn("req-1", "ep-1", threshold="high")
    with pytest.raises(ValueError, match="action must be one of"):
        cls.spawn("req-1", "ep-1", action="explode")


def test_extra_kwargs_forwarded_to_function(ctx):
    @probe(register=False, threshold=1.0)
    def scaled(record, *, scale=1.0, offset):
        return record.tensor[0] * scale + offset

    signals, _ = _run(scaled, [0.4], ctx, scale=2.0, offset=0.2)
    assert signals[0].confidence == pytest.approx(1.0)
    assert signals[0].action is ProbeAction.ABORT


def test_var_keyword_function_accepts_any_kwargs(ctx):
    @probe(register=False)
    def anything(record, **options):
        return float(options["k"])

    signals, _ = _run(anything, [0.0], ctx, k=3)
    assert signals[0].confidence == 3.0


def test_unknown_kwarg_rejected_at_spawn():
    @probe("kw", register=False)
    def scorer(record, *, scale=1.0):
        return 0.0

    with pytest.raises(TypeError, match=r"unexpected argument\(s\) bogus.*accepts: action, scale, threshold"):
        scorer.spawn("req-1", "ep-1", bogus=1)


def test_missing_required_kwarg_rejected_at_spawn():
    @probe("req", register=False)
    def scorer(record, *, weights):
        return 0.0

    with pytest.raises(TypeError, match=r"requires keyword argument.*weights"):
        scorer.spawn("req-1", "ep-1")


def test_probe_factory(ctx):
    cls = probe("pf", register=False)(_score)
    factory = ProbeFactory(cls, {"threshold": 0.9})
    instance = factory.spawn("req-1", "ep-1", threshold=0.3)
    assert instance.on_activation(_record(0.5)).action is ProbeAction.ABORT
    instance = factory.spawn("req-1", "ep-1")
    assert instance.on_activation(_record(0.5)).action is ProbeAction.CONTINUE


# -- Router end-to-end -------------------------------------------------------


def _inline_point(probe_type, **probe_args):
    point = make_extraction_point(name="ep-1", probe_type=probe_type, probe_kind=ProbeKind.SINGLE_SHOT)
    return replace(point, probe_args=probe_args)


def _route_until_abort(router, values):
    """A minimal engine loop: route one activation per token, stop on ABORT."""
    routed = 0
    for i, value in enumerate(values):
        signal = router.route(_record(value, token_pos=i))
        routed += 1
        if signal is not None and signal.action is ProbeAction.ABORT:
            break
    return routed


def test_router_inline_abort_stops_generation(registry, ctx):
    probe("tox", threshold=0.8, registry=registry)(_score)
    router = Router(registry)

    with router.request([_inline_point("tox")], request_id="req-1") as handle:
        routed = _route_until_abort(router, [0.1, 0.5, 0.85, 0.2, 0.1])
    results = handle.results

    assert routed == 3
    assert results["ep-1"].verdict == {"flagged": True, "max_score": 0.85, "n": 3}
    assert [s.action for s in results["ep-1"].signal_history] == [
        ProbeAction.CONTINUE,
        ProbeAction.CONTINUE,
        ProbeAction.ABORT,
    ]


def test_router_probe_args_override_threshold(registry, ctx):
    probe("tox", threshold=0.8, registry=registry)(_score)
    router = Router(registry)

    with router.request([_inline_point("tox", threshold=0.4)], request_id="req-1") as handle:
        routed = _route_until_abort(router, [0.1, 0.5, 0.85])

    assert routed == 2
    assert handle.results["ep-1"].verdict["flagged"] is True


def test_router_action_flag_does_not_stop_generation(registry, ctx):
    probe("tox", threshold=0.8, registry=registry)(_score)
    router = Router(registry)

    with router.request([_inline_point("tox", action="flag")], request_id="req-1") as handle:
        routed = _route_until_abort(router, [0.9, 0.9, 0.1])

    assert routed == 3
    assert handle.results["ep-1"].verdict == {"flagged": True, "max_score": 0.9, "n": 3}


def test_router_rejects_function_probe_at_async_point(registry, ctx):
    """Function probes are single_shot, and async points need trajectory
    probes, so a function probe can never run (or abort) off the generation
    path: the Router refuses the wiring up front."""
    probe("tox", threshold=0.8, registry=registry)(_score)
    router = Router(registry)

    async_single = make_extraction_point(
        name="ep-1", probe_type="tox", probe_kind=ProbeKind.SINGLE_SHOT, execution_mode=ExecutionMode.ASYNC
    )
    with pytest.raises(RouterError, match="async is not valid with probe_kind=single_shot"):
        router.register_request("req-1", [async_single], ctx)

    async_trajectory = make_extraction_point(
        name="ep-1", probe_type="tox", probe_kind=ProbeKind.TRAJECTORY, execution_mode=ExecutionMode.ASYNC
    )
    with pytest.raises(RouterError, match="implements probe_kind='single_shot'"):
        router.register_request("req-1", [async_trajectory], ctx)


# -- equivalence with a hand-written class -----------------------------------


class _HandWrittenThresholdProbe(Probe):
    """What @probe("tox", threshold=0.5) on _score is shorthand for."""

    probe_kind = "single_shot"

    def __init__(self, threshold=0.5):
        super().__init__()
        self._threshold = threshold
        self._history = []
        self._max_score = None

    def on_start(self, request_ctx):
        pass

    def on_activation(self, record):
        score = float(record.tensor[0])
        self._max_score = score if self._max_score is None else max(self._max_score, score)
        action = ProbeAction.ABORT if score >= self._threshold else ProbeAction.CONTINUE
        signal = ProbeSignal(action=action, confidence=score, metadata={"score": score, "probe": "tox"})
        self._history.append(signal)
        return signal

    def on_end(self, request_ctx):
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={
                "flagged": any(s.action is ProbeAction.ABORT for s in self._history),
                "max_score": self._max_score,
                "n": len(self._history),
            },
            signal_history=list(self._history),
        )


def _strip_timestamp(signal):
    return None if signal is None else (signal.action, signal.confidence, signal.metadata)


def test_function_probe_matches_hand_written_class(registry, ctx):
    probe("tox", threshold=0.5, registry=registry)(_score)
    registry.register("tox_class", _HandWrittenThresholdProbe)
    router = Router(registry)
    point_fn = _inline_point("tox")
    point_cls = replace(point_fn, name="ep-2", probe_type="tox_class")
    values = [0.1, 0.5, 0.3, 0.9]

    signals_fn, signals_cls = [], []
    with router.request([point_fn, point_cls], request_id="req-1") as handle:
        for i, value in enumerate(values):
            signals_fn.append(router.route(_record(value, i, extraction_point_name="ep-1")))
            signals_cls.append(router.route(_record(value, i, extraction_point_name="ep-2")))
    fn_result, cls_result = handle.results["ep-1"], handle.results["ep-2"]

    assert [_strip_timestamp(s) for s in signals_fn] == [_strip_timestamp(s) for s in signals_cls]
    assert fn_result.verdict == cls_result.verdict == {"flagged": True, "max_score": 0.9, "n": 4}
    assert [_strip_timestamp(s) for s in fn_result.signal_history] == [
        _strip_timestamp(s) for s in cls_result.signal_history
    ]
    assert fn_result.metadata == cls_result.metadata

"""Function probes: write a stateless single_shot probe as a plain function.

Most probes answer one question per activation ("how toxic is this?") and
keep no state between activations. For those, the four-method
`Probe` lifecycle is boilerplate. The
`probe()` decorator turns a function ``(record) -> score`` into an
ordinary ``Probe`` subclass, registered by name::

    import torch
    from undercurrent.core.function_probe import probe

    head = torch.nn.Linear(768, 1)  # your trained probe head

    @probe("toxicity", threshold=0.8)
    def toxicity(record) -> float:
        return head(record.tensor.float()).sigmoid().item()

    # toxicity is a Probe subclass: use it in ProbedModel(probes={...}),
    # ProbeFactory(toxicity, {...}), or by name ("toxicity") from a spec.
    toxicity.fn(record)  # the original function, for unit tests

Return values, per activation:

- ``None``: nothing to report. No signal, not counted.
- ``bool``: ``True`` flags the activation, ``False`` doesn't.
- ``float`` / ``int``: a score. Flagged iff a ``threshold`` is set and
  ``score >= threshold``; with no threshold the score is only recorded.
- `ProbeSignal`: passed through untouched,
  for full control over action, confidence and metadata.

A flagged activation emits ``ProbeSignal(action=<action>, confidence=<score,
or 1.0 for True>, metadata={"score": <score or None>, "probe": <name>})``.
``action`` is ``"abort"`` (the default; an inline extraction point stops
generation) or ``"flag"`` (generation continues, the signal is recorded).
Every other non-``None`` return emits a ``CONTINUE`` signal with the same
metadata, so scores reach ``signal_history``, sinks and
``GenerationOutput.probe_results``.

``on_end`` returns a `ProbeResult` whose
verdict is ``{"flagged": bool, "max_score": float | None, "n": int}``:
whether any activation was flagged, the highest numeric score seen, and how
many activations returned something other than ``None``.

``threshold`` and ``action`` can be overridden per extraction point through
``probe_args`` in the spec, or through ``ProbeFactory`` kwargs. Any other
kwargs are passed to the function as keyword arguments, so declare them as
keyword-only parameters::

    @probe("norm", threshold=10.0)
    def norm(record, *, ord: int = 2) -> float:
        return float(record.tensor.float().norm(p=ord))

Function probes are ``probe_kind = "single_shot"``. The spec only allows
``single_shot`` at inline extraction points, so a function probe always
runs inline, on the generation path, and its ``ABORT`` signals take effect.
For async (observe-only) points, where the Router treats signals as
observational, or for anything that keeps state across activations, write a
``trajectory`` ``Probe`` subclass instead.

Exceptions raised by the function propagate out of ``on_activation``
exactly like a class probe's, and the Router handles them the same way.
"""

from __future__ import annotations

import inspect
import numbers
from collections.abc import Callable
from typing import Any, overload

from ..errors import ProbeDefinitionError, ProbingTypeError, ProbingValueError, did_you_mean
from .activation import ActivationRecord
from .context import RequestContext
from .probe import Probe
from .registry import ProbeRegistry, default_registry
from .result import ProbeResult
from .signal import ProbeAction, ProbeSignal

#: Values accepted for ``action``, mapped to the signal a flagged activation emits.
_ACTIONS: dict[str, ProbeAction] = {"abort": ProbeAction.ABORT, "flag": ProbeAction.FLAG}

_ALLOWED_RETURNS = "float, int, bool, ProbeSignal or None"

ProbeFunction = Callable[..., Any]


def _check_threshold(threshold: Any, where: str) -> float | None:
    if threshold is None:
        return None
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ProbingTypeError(f"{where}: threshold must be a number or None (got {threshold!r}), e.g. threshold=0.5")
    return float(threshold)


def _check_action(action: Any, where: str) -> ProbeAction:
    if isinstance(action, ProbeAction) and action in _ACTIONS.values():
        return action
    if isinstance(action, str) and action in _ACTIONS:
        return _ACTIONS[action]
    raise ProbingValueError(
        f"{where}: action must be one of {sorted(_ACTIONS)}, got {action!r}.{did_you_mean(action, _ACTIONS)}"
    )


def _inspect_signature(fn: ProbeFunction) -> tuple[frozenset[str], frozenset[str], bool]:
    """Check ``fn`` takes one positional ``record`` plus keyword-only params.

    Returns (accepted keyword names, required keyword names, accepts **kwargs).
    """
    name = getattr(fn, "__qualname__", repr(fn))
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ProbeDefinitionError(
            f"@probe: cannot inspect the signature of {name}: {exc}. Decorate a plain Python function "
            "`def fn(record, *, option=...)`."
        ) from exc

    positional = []
    keyword: set[str] = set()
    required: set[str] = set()
    var_keyword = False
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            positional.append(param.name)
        elif param.kind is param.VAR_POSITIONAL:
            raise ProbeDefinitionError(
                f"@probe: {name} takes *{param.name}; a function probe is called with exactly one "
                f"positional argument, the ActivationRecord. Use `def {fn.__name__}(record, *, option=...)`."
            )
        elif param.kind is param.KEYWORD_ONLY:
            keyword.add(param.name)
            if param.default is param.empty:
                required.add(param.name)
        else:
            var_keyword = True

    if len(positional) != 1:
        raise ProbeDefinitionError(
            f"@probe: {name} must take exactly one positional parameter (the ActivationRecord), "
            f"got {len(positional)} ({', '.join(positional) or 'none'}). Make extra options keyword-only: "
            f"`def {getattr(fn, '__name__', 'fn')}(record, *, option=...)`."
        )
    reserved = keyword & {"threshold", "action"}
    if reserved:
        raise ProbeDefinitionError(
            f"@probe: {name} declares {', '.join(sorted(reserved))} as a parameter, but @probe handles "
            f"threshold and action itself and never passes them to the function. Rename the parameter."
        )
    return frozenset(keyword), frozenset(required), var_keyword


class FunctionProbe(Probe):
    """Base class of every class [`@probe`][undercurrent.core.function_probe.probe] creates. Not used directly.

    The decorator fills in the class attributes below. Instances hold the
    per-request state: the effective threshold/action, the extra kwargs for
    the function, and the signal history.

    Attributes:
        fn: the decorated function, unchanged.
        probe_name: the ``probe_type`` name the class was created with.
        default_threshold: the decorator's ``threshold``.
        default_action: the decorator's ``action``, as a ``ProbeAction``.
    """

    probe_kind = "single_shot"

    # Set on each generated subclass by @probe.
    fn: ProbeFunction
    probe_name: str
    default_threshold: float | None = None
    default_action: ProbeAction = ProbeAction.ABORT
    _accepted_kwargs: frozenset[str] = frozenset()
    _required_kwargs: frozenset[str] = frozenset()
    _accepts_var_kwargs: bool = False

    def __init__(self, *, threshold: Any = ..., action: Any = ..., **fn_kwargs: Any) -> None:
        super().__init__()
        cls = type(self)
        where = f"probe {cls.probe_name!r}"
        self.threshold = cls.default_threshold if threshold is ... else _check_threshold(threshold, where)
        self.action = cls.default_action if action is ... else _check_action(action, where)

        if not cls._accepts_var_kwargs:
            unknown = sorted(set(fn_kwargs) - cls._accepted_kwargs)
            if unknown:
                accepted = ", ".join(sorted(cls._accepted_kwargs | {"threshold", "action"}))
                hint = did_you_mean(unknown[0], cls._accepted_kwargs | {"threshold", "action"})
                raise ProbingTypeError(
                    f"{where}: unexpected argument(s) {', '.join(unknown)} (from probe_args or ProbeFactory "
                    f"kwargs).{hint} {cls.fn.__qualname__} accepts: {accepted}. Add a keyword-only parameter "
                    f"to the function to accept it."
                )
        missing = sorted(cls._required_kwargs - set(fn_kwargs))
        if missing:
            raise ProbingTypeError(
                f"{where}: {cls.fn.__qualname__} requires keyword argument(s) {', '.join(missing)}; "
                f"set them in the extraction point's probe_args or the ProbeFactory kwargs."
            )
        self.fn_kwargs = dict(fn_kwargs)
        self._signal_history: list[ProbeSignal] = []
        self._flagged = False
        self._max_score: float | None = None
        self._n = 0

    def on_start(self, request_ctx: RequestContext) -> None:
        pass

    def on_activation(self, record: ActivationRecord) -> ProbeSignal | None:
        value = type(self).fn(record, **self.fn_kwargs)
        signal = self._to_signal(value)
        if signal is None:
            return None
        self._n += 1
        if signal.action is not ProbeAction.CONTINUE:
            self._flagged = True
        self._signal_history.append(signal)
        return signal

    def _to_signal(self, value: Any) -> ProbeSignal | None:
        if value is None:
            return None
        if isinstance(value, ProbeSignal):
            return value
        name = type(self).probe_name
        if isinstance(value, bool):
            action = self.action if value else ProbeAction.CONTINUE
            return ProbeSignal(
                action=action, confidence=1.0 if value else None, metadata={"score": None, "probe": name}
            )
        if isinstance(value, numbers.Real):
            score = float(value)
            self._max_score = score if self._max_score is None else max(self._max_score, score)
            flagged = self.threshold is not None and score >= self.threshold
            action = self.action if flagged else ProbeAction.CONTINUE
            return ProbeSignal(action=action, confidence=score, metadata={"score": score, "probe": name})
        raise ProbingTypeError(
            f"probe {name!r}: {type(self).fn.__qualname__} returned {type(value).__name__} ({value!r:.80}); "
            f"a function probe must return {_ALLOWED_RETURNS}. Convert tensors with float(...) or .item()."
        )

    def on_end(self, request_ctx: RequestContext) -> ProbeResult:
        return ProbeResult(
            request_id=request_ctx.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"flagged": self._flagged, "max_score": self._max_score, "n": self._n},
            signal_history=list(self._signal_history),
        )


@overload
def probe(name: ProbeFunction, /) -> type[FunctionProbe]: ...


@overload
def probe(
    name: str | None = None,
    /,
    *,
    threshold: float | None = None,
    action: str = "abort",
    registry: ProbeRegistry | None = None,
    register: bool = True,
) -> Callable[[ProbeFunction], type[FunctionProbe]]: ...


def probe(
    name: str | ProbeFunction | None = None,
    /,
    *,
    threshold: float | None = None,
    action: str = "abort",
    registry: ProbeRegistry | None = None,
    register: bool = True,
) -> Any:
    """Turn ``fn(record) -> float | bool | ProbeSignal | None`` into a single_shot Probe.

    ```python
    @probe("norm", threshold=10.0)
    def norm(record, *, ord: int = 2) -> float:
        return float(record.tensor.float().norm(p=ord))
    ```

    What the function returns, per activation:

    - ``None``: nothing to report. No signal, not counted.
    - ``bool``: ``True`` flags the activation, ``False`` doesn't.
    - ``float`` / ``int``: a score. Flagged if a ``threshold`` is set and
      ``score >= threshold``; with no threshold the score is only recorded.
    - a [`ProbeSignal`][undercurrent.core.ProbeSignal]: passed through
      untouched, for full control over action, confidence and metadata.

    A flagged activation emits a signal with the probe's ``action``; every
    other non-``None`` return emits a ``CONTINUE`` signal, so scores reach
    ``signal_history`` and sinks. The verdict is
    ``{"flagged": bool, "max_score": float | None, "n": int}``.

    ``threshold`` and ``action`` can be overridden per extraction point
    through ``probe_args`` in the spec. Any other ``probe_args`` are passed
    to the function as keyword arguments, so declare them keyword-only.
    Function probes are single_shot, so they always run inline; write a
    [`Probe`][undercurrent.core.Probe] subclass for anything that keeps
    state across activations.

    Args:
        name: the ``probe_type`` to register under. Defaults to the
            function's ``__name__``. ``@probe`` without parentheses also works.
        threshold: a numeric score ``>= threshold`` flags the activation.
            ``None`` (the default) records scores without flagging.
        action: what a flagged activation asks for: ``"abort"`` (default)
            stops generation at an inline extraction point, ``"flag"``
            only records it.
        registry: where to register. Defaults to the global registry
            ([`default_registry`][undercurrent.core.default_registry]).
        register: set False to create the class without registering it.

    Returns:
        A [`FunctionProbe`][undercurrent.core.FunctionProbe] subclass
            named after the function. The function stays available as
            ``Cls.fn``, for unit tests.
    """
    if callable(name):
        return probe()(name)
    if name is not None and (not isinstance(name, str) or not name):
        raise ProbingValueError(
            f"@probe: name must be a non-empty string or None (got {name!r}). Use @probe('my_probe'), or "
            "plain @probe to register under the function's name."
        )
    default_threshold = _check_threshold(threshold, "@probe")
    default_action = _check_action(action, "@probe")

    def decorator(fn: ProbeFunction) -> type[FunctionProbe]:
        if isinstance(fn, type) or not callable(fn):
            raise ProbeDefinitionError(
                f"@probe decorates a function `def fn(record) -> score`, got {fn!r}. "
                "For a class, subclass Probe and use @register_probe('name') instead."
            )
        accepted, required, var_kw = _inspect_signature(fn)
        probe_name = name if name is not None else fn.__name__
        cls = type(
            fn.__name__,
            (FunctionProbe,),
            {
                "__module__": fn.__module__,
                "__qualname__": fn.__qualname__,
                "__doc__": fn.__doc__,
                "fn": staticmethod(fn),
                "probe_name": probe_name,
                "default_threshold": default_threshold,
                "default_action": default_action,
                "_accepted_kwargs": accepted,
                "_required_kwargs": required,
                "_accepts_var_kwargs": var_kw,
            },
        )
        if register:
            (registry if registry is not None else default_registry).register(probe_name, cls)
        return cls

    return decorator


__all__ = ["FunctionProbe", "probe"]

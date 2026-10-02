"""RequestHandle: the context manager `Router.request(...)` returns.

Wraps one request's `register_request` -> `route`* -> `end_request`
lifecycle so `end_request` always runs (even when the body raises) and the
request's results stay available after the block::

    with router.request(extraction_points=spec, prompt_metadata={"prompt": p}) as req:
        for record in stream:
            sig = req.route(record)
            if sig is not None and sig.action is ProbeAction.ABORT:
                break
    req.results   # {extraction_point_name: ProbeResult}
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any

from ..core import ProbeResult, ProbeSignal, RequestContext
from ..spec import ActivationRecord, ExtractionPoint
from .errors import RouterError

if TYPE_CHECKING:
    from .router import Router

logger = logging.getLogger(__name__)


class RequestHandle:
    """One request's lifecycle on a `Router`. Create it via `Router.request`.

    `__enter__` registers the request; `__exit__` calls `end_request`,
    whether or not the body raised. If the body raised and `end_request`
    then also raises, the `end_request` error is logged and the body's
    exception propagates unchanged -- cleanup never masks the real error.

    Single use: a handle can be entered once.
    """

    def __init__(
        self,
        router: Router,
        request_id: str,
        extraction_points: list[ExtractionPoint],
        prompt_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._router = router
        self._request_id = request_id
        self._extraction_points = extraction_points
        self._request_ctx = RequestContext(
            request_id=request_id,
            prompt_metadata=dict(prompt_metadata or {}),
            extraction_point_config=None,
        )
        self._entered = False
        self._ended = False
        self._results: dict[str, ProbeResult] | None = None

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def request_ctx(self) -> RequestContext:
        return self._request_ctx

    @property
    def ended(self) -> bool:
        """True once `end_request` has been attempted for this request."""
        return self._ended

    @property
    def results(self) -> dict[str, ProbeResult]:
        """`{extraction_point_name: ProbeResult}`, available after the `with` block.

        Raises `RouterError` while the request is still active, or if
        `end_request` failed and so produced no results.
        """
        if self._results is None:
            if not self._ended:
                raise RouterError(
                    f"results for request_id={self._request_id!r} are only available after the request ends "
                    "(read them after the `with router.request(...)` block)"
                )
            raise RouterError(
                f"end_request failed for request_id={self._request_id!r}, so no results are available; "
                "see the error logged by the 'undercurrent.router' logger"
            )
        return self._results

    def route(self, record: ActivationRecord) -> ProbeSignal | None:
        """Same as `Router.route`, but first checks that `record` belongs to this request."""
        if record.request_id != self._request_id:
            raise RouterError(
                f"record.request_id={record.request_id!r} does not match this request handle's "
                f"request_id={self._request_id!r}. Route each record through the handle of its own request."
            )
        if not self._entered or self._ended:
            raise RouterError(
                f"request_id={self._request_id!r} is not active; route() is only valid inside its `with` block"
            )
        return self._router.route(record)

    def __enter__(self) -> RequestHandle:
        if self._entered:
            raise RouterError(
                f"request handle for request_id={self._request_id!r} can only be entered once; "
                "create a new one with router.request(...)"
            )
        self._router.register_request(self._request_id, self._extraction_points, self._request_ctx)
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._ended = True
        if exc is None:
            self._results = self._router.end_request(self._request_id)
            return
        try:
            self._results = self._router.end_request(self._request_id)
        except Exception:  # noqa: BLE001 -- must not mask the body's exception, which propagates below
            logger.exception(
                "end_request(%r) raised while handling an exception from the request body; "
                "the body's exception is re-raised",
                self._request_id,
            )

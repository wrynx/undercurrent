"""Redaction hooks: control exactly what probe-derived data a sink lets out.

Every built-in sink takes a ``redact`` keyword: a function
``(record: dict) -> dict | None`` applied to the fully serialised, JSON-able
record just before it's written, sent or dead-lettered. Returning ``None``
drops the record. The function gets a deep copy, so it may mutate its
argument freely without touching the caller's objects. If it raises, the
record is dropped (fail closed, never sent unredacted) and the sink's
``redaction_error_count`` goes up.

Helpers::

    redact_keys("prompt", "text")            # replace values with "[REDACTED]"
    drop_keys("metadata.raw_scores")         # remove the key entirely
    chain(drop_keys("debug"), redact_keys("prompt"))

A key pattern without a dot matches that key at any depth. A dotted pattern
such as ``"metadata.raw_scores"`` matches a ``raw_scores`` key whose parent
key is ``metadata``, wherever that pair sits in the record. Only dict keys
count as path segments; lists are walked through transparently, so
``"signal_history.metadata.prompt"`` also matches inside every element of the
``signal_history`` list. Matching is exact and case-sensitive (``prompt``
does not match ``prompt_len``).

Field audit: where user text can appear in a record
---------------------------------------------------

Records are built by ``records.build_log_record``::

    {"kind", "request_id", "extraction_point_name", "timestamp", "payload"}

- ``kind``, ``timestamp``: library-generated, never user text.
- ``request_id``: chosen by the caller of ``Router.register_request`` (or the
  adapter). Normally an opaque id, but a caller could put anything in it.
- ``extraction_point_name``: from the spec, operator-controlled.
- ``payload`` for a signal (``ProbeSignal``): ``action``, ``confidence`` and
  ``timestamp`` are scalars; **``metadata`` is probe-defined and may hold
  anything**.
- ``payload`` for a result (``ProbeResult``): ``request_id`` and
  ``extraction_point_name`` as above; **``verdict`` and ``metadata`` are
  probe-defined**; ``signal_history`` is a list of signal payloads (same
  rules as above).
- Tensors and bytes are never dumped: ``serialize.to_jsonable`` summarises
  them to shape/dtype/length.

The library itself never writes prompt or generated text into a record. The
risk is a probe copying it in: the HF and vLLM adapters hand every probe
``RequestContext.prompt_metadata = {"model", "prompt", "prompt_len"}`` in
``on_start``, so ``prompt`` (the raw prompt string) is one assignment away from
``ProbeSignal.metadata`` / ``ProbeResult.metadata`` / ``verdict``. Probes that
decode tokens commonly use keys like ``text``, ``generated_text`` or
``completion``, and token ids (``input_ids``, ``token_ids``) are text in another
encoding. One router-generated field can also echo user data: when a probe's
``on_activation`` raises, the router logs a synthetic signal whose
``metadata["error"]`` is ``str(exc)``, i.e. whatever the probe put in its
exception message. It is not redacted by default (it's what operators need
for debugging); add ``redact_keys("error")`` if your probes may echo input.

`DEFAULT_PROMPT_TEXT_KEYS` lists the keys ``WebhookLogSink`` redacts by
default (pass ``include_prompt_text=True`` to opt out). ``FileLogSink`` does not
redact unless given a ``redact`` function, since the data stays on the machine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import ProbingTypeError, ProbingValueError

RedactFn = Callable[[dict[str, Any]], dict[str, Any] | None]
"""A redaction hook: takes a JSON-able record, returns it (possibly changed) or ``None`` to drop it."""

DEFAULT_PROMPT_TEXT_KEYS: tuple[str, ...] = (
    "prompt",
    "prompts",
    "prompt_text",
    "text",
    "input_text",
    "output_text",
    "generated_text",
    "completion",
    "response",
    "messages",
    "input_ids",
    "output_ids",
    "token_ids",
)
"""Keys that can carry prompt or generated text, redacted by ``WebhookLogSink`` by default."""

DEFAULT_REPLACEMENT = "[REDACTED]"

_DROP = object()


def _parse_patterns(keys: tuple[str, ...], fn_name: str) -> tuple[tuple[str, ...], ...]:
    if not keys:
        raise ProbingValueError(f"{fn_name}() needs at least one key, e.g. {fn_name}('prompt', 'metadata.user_id')")
    patterns = []
    for key in keys:
        if not isinstance(key, str):
            raise ProbingTypeError(
                f"{fn_name}() keys must be strings (got {type(key).__name__} {key!r}), e.g. {fn_name}('prompt')"
            )
        segments = tuple(key.split("."))
        if any(not s for s in segments):
            raise ProbingValueError(
                f"{fn_name}(): invalid key pattern {key!r}: empty segment. Use a key name or a dotted path "
                "such as 'prompt' or 'metadata.user_id' (no leading, trailing or doubled dots)."
            )
        patterns.append(segments)
    return tuple(patterns)


def _matches(path: tuple[str, ...], patterns: tuple[tuple[str, ...], ...]) -> bool:
    return any(len(path) >= len(p) and path[-len(p) :] == p for p in patterns)


def _transform(value: Any, path: tuple[str, ...], patterns: tuple[tuple[str, ...], ...], replacement: Any) -> Any:
    """Return a copy of `value` with every matching key replaced (or dropped, if `replacement is _DROP`)."""
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            child_path = (*path, str(key))
            if _matches(child_path, patterns):
                if replacement is not _DROP:
                    out[key] = replacement
            else:
                out[key] = _transform(child, child_path, patterns, replacement)
        return out
    if isinstance(value, (list, tuple)):
        return [_transform(item, path, patterns, replacement) for item in value]
    return value


def redact_keys(*keys: str, replacement: Any = DEFAULT_REPLACEMENT) -> RedactFn:
    """Replace the value of every matching key with `replacement`.

    `keys` are plain key names (matched at any depth) or dotted paths
    (``"payload.metadata.prompt"``, matched wherever that chain of keys
    appears). The key itself stays in the record, so consumers can see that
    something was removed.
    """
    patterns = _parse_patterns(keys, "redact_keys")

    def _redact(record: dict[str, Any]) -> dict[str, Any]:
        redacted: dict[str, Any] = _transform(record, (), patterns, replacement)
        return redacted

    return _redact


def drop_keys(*keys: str) -> RedactFn:
    """Remove every matching key (and its value) from the record.

    Key matching works exactly as in [`redact_keys`][undercurrent.sinks.redact_keys].
    """
    patterns = _parse_patterns(keys, "drop_keys")

    def _drop(record: dict[str, Any]) -> dict[str, Any]:
        dropped: dict[str, Any] = _transform(record, (), patterns, _DROP)
        return dropped

    return _drop


def chain(*fns: RedactFn) -> RedactFn:
    """Compose redaction functions left to right. If any returns ``None``, the record is dropped."""

    def _chained(record: dict[str, Any]) -> dict[str, Any] | None:
        current = record
        for fn in fns:
            result = fn(current)
            if result is None:
                return None
            current = result
        return current

    return _chained

"""Entry points for turning YAML/dict specs into resolved `ProbeSpec` objects.

This is the module most callers should use:

    from undercurrent.spec import load_yaml_file
    spec = load_yaml_file("my_probes.yaml")
    for extraction_point in spec:
        ...

Every public function here raises `SpecValidationError`
(never a raw pydantic or PyYAML exception) on invalid input, so callers only
need to handle one exception type. Each problem is reported with where it is
(``extraction_points[2] 'drift'.position``), the YAML line when the spec came
from YAML text, and the file path when it came from `load_yaml_file()`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic import ValidationError as _PydanticValidationError

from ..errors import ProbingTypeError, SpecFileNotFoundError, did_you_mean, one_of
from .errors import PositionSyntaxError, SpecValidationError
from .position import parse_position
from .resolved import ExtractionPoint, InterventionPolicy, ProbeSpec
from .schema import (
    DEPRECATED_KEY_ALIASES,
    ExecutionMode,
    ExtractionPointSpec,
    InterventionMode,
    InterventionPolicySpec,
    ProbeKind,
    ProbeSpecFile,
    TensorType,
    TimeoutAction,
    UntilKind,
)

SpecSource = str | os.PathLike[str] | Mapping[str, Any] | ProbeSpec
"""Everything `load_spec()` accepts: a YAML file path, YAML text, a dict or a `ProbeSpec`."""


def load_spec(spec: SpecSource) -> ProbeSpec:
    """Load a spec from whatever form you have it in.

    - a [`ProbeSpec`][undercurrent.spec.ProbeSpec] is returned unchanged;
    - an ``os.PathLike`` (e.g. ``pathlib.Path``) is read with
      [`load_yaml_file`][undercurrent.spec.load_yaml_file];
    - a ``str`` is a file path if it is one line and names an existing file or
      ends in ``.yaml``/``.yml``/``.json`` (read with ``load_yaml_file``);
      otherwise it is YAML text (parsed with
      [`parse_yaml`][undercurrent.spec.parse_yaml]);
    - a mapping is validated with [`parse_dict`][undercurrent.spec.parse_dict].

    ```python
    from undercurrent import load_spec

    spec = load_spec("probes.yaml")
    spec = load_spec(
        "extraction_points: [{name: p, layers: 0, tensor_type: residual_stream,"
        " position: 'prompt[-1]', probe_type: norm, probe_kind: single_shot}]"
    )
    ```

    Raises:
        SpecValidationError: the spec is invalid.
        SpecFileNotFoundError: the file doesn't exist.
        ProbingTypeError: ``spec`` is of an unsupported type (a ``TypeError``).
    """
    if isinstance(spec, ProbeSpec):
        return spec
    if isinstance(spec, os.PathLike):
        return load_yaml_file(spec)
    if isinstance(spec, str):
        if _looks_like_path(spec):
            return load_yaml_file(spec)
        return parse_yaml(spec)
    if isinstance(spec, Mapping):
        return parse_dict(dict(spec))
    raise ProbingTypeError(
        f"load_spec() takes a YAML file path, a YAML string, a dict or a ProbeSpec, got {type(spec).__name__}"
    )


def _looks_like_path(text: str) -> bool:
    """A one-line string naming an existing file or ending in .yaml/.yml/.json
    is a path; anything else is YAML text."""
    if "\n" in text:
        return False
    return os.path.isfile(text) or text.lower().endswith((".yaml", ".yml", ".json"))


def parse_dict(data: dict[str, Any]) -> ProbeSpec:
    """Validate a plain dict (already loaded from YAML/JSON) and resolve it.

    Raises:
        SpecValidationError: any structural or semantic problem.
    """
    return _parse_data(data, node=None, source=None)


def parse_yaml(text: str) -> ProbeSpec:
    """Parse and resolve a spec from a YAML (or JSON, which is valid YAML) string."""
    return _parse_text(text, source=None)


def load_yaml_file(path: str | os.PathLike[str]) -> ProbeSpec:
    """Load, parse, and resolve a spec from a YAML file on disk."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError as exc:
        raise SpecFileNotFoundError(
            f"spec file {os.fspath(path)!r} not found (looked relative to {os.getcwd()!r}). "
            "Check the path, or pass the spec as YAML text or a dict instead."
        ) from exc
    return _parse_text(text, source=os.fspath(path))


def _parse_text(text: str, source: str | None) -> ProbeSpec:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        line = None
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            line = mark.line + 1
        issue = _render(source, line, "", f"invalid YAML: {_describe_yaml_error(exc)}")
        raise SpecValidationError(issue) from exc

    if not isinstance(data, dict):
        got = "an empty document" if data is None else f"a {type(data).__name__}"
        issue = _render(
            source,
            None,
            "",
            f"expected a YAML mapping with an 'extraction_points' key at the top level, got {got}. "
            "Start the spec with `extraction_points:` followed by a list of extraction points.",
        )
        raise SpecValidationError(issue)

    try:
        node = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:  # pragma: no cover -- safe_load just accepted the same text
        node = None
    return _parse_data(data, node=node, source=source)


def _parse_data(data: Any, *, node: yaml.Node | None, source: str | None) -> ProbeSpec:
    try:
        raw = ProbeSpecFile.model_validate(data)
    except _PydanticValidationError as exc:
        issues = [_render(source, issue.line, issue.where, issue.message) for issue in _issues(exc, data, node)]
        raise SpecValidationError(_combine(issues), issues=issues) from exc
    except PositionSyntaxError as exc:
        # parse_position raises this from inside a model validator; pydantic
        # doesn't wrap non-ValueErrors, so recover which point it was.
        index = _bad_position_index(data)
        where, line = "", None
        if index is not None:
            loc = ("extraction_points", index, "position")
            where = _where(loc, data)
            line = _line(node, loc)
        issue = _render(source, line, where, str(exc))
        raise SpecValidationError(issue) from exc

    return _resolve(raw)


def _resolve(raw: ProbeSpecFile) -> ProbeSpec:
    return ProbeSpec(
        version=raw.version,
        extraction_points=tuple(_resolve_point(p) for p in raw.extraction_points),
    )


def _resolve_point(p: ExtractionPointSpec) -> ExtractionPoint:
    layers = (p.layers,) if isinstance(p.layers, int) else tuple(p.layers)
    return ExtractionPoint(
        name=p.name,
        layers=layers,
        tensor_type=p.tensor_type,
        position=parse_position(p.position),
        stride=p.stride,
        until=p.until,
        probe_type=p.probe_type,
        probe_kind=p.probe_kind,
        execution_mode=p.execution_mode,
        queue_depth=p.queue_depth,
        intervention=_resolve_intervention(p.intervention),
        probe_args=p.probe_args,
    )


def _resolve_intervention(spec: InterventionPolicySpec | None) -> InterventionPolicy | None:
    if spec is None:
        return None
    return InterventionPolicy(mode=spec.mode, timeout_ms=spec.timeout_ms, on_timeout=spec.on_timeout)


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

#: Spec fields whose value must be one of an Enum's values.
_ENUM_FIELDS: dict[str, type[Enum]] = {
    "tensor_type": TensorType,
    "probe_kind": ProbeKind,
    "execution_mode": ExecutionMode,
    "until": UntilKind,
    "mode": InterventionMode,
    "on_timeout": TimeoutAction,
}

#: Which pydantic model validates the mapping found under a key.
_NESTED_MODELS: dict[tuple[type[BaseModel], str], type[BaseModel]] = {
    (ProbeSpecFile, "extraction_points"): ExtractionPointSpec,
    (ExtractionPointSpec, "intervention"): InterventionPolicySpec,
}

#: Other spellings a key may have in the YAML (aliases and deprecated keys).
_KEY_SPELLINGS: dict[str, tuple[str, ...]] = {
    "stride": ("every_n",),
    **{new: (old,) for old, new in DEPRECATED_KEY_ALIASES.items()},
}

#: How pydantic names the branches of a union type in an error ``loc``.
_UNION_BRANCHES = {"int": "an int", "str": "a string", "list[int]": "a list of ints"}

_POINT_PREFIX = re.compile(r"^extraction point '(?P<name>[^']*)': ")


@dataclass
class _Issue:
    where: str
    message: str
    line: int | None
    #: The loc up to (not including) a union-branch tag, and that tag.
    field_loc: tuple[Any, ...] = ()
    branch: str | None = None
    error_type: str = ""
    got: str = ""


def _render(source: str | None, line: int | None, where: str, message: str) -> str:
    """``probes.yaml:14: extraction_points[2] 'drift'.position: message``."""
    if source is not None:
        prefix = f"{source}:{line}" if line is not None else source
    else:
        prefix = f"line {line}" if line is not None else ""
    return ": ".join(part for part in (prefix, where, message) if part)


def _combine(issues: list[str]) -> str:
    if len(issues) == 1:
        return issues[0]
    return f"{len(issues)} problems in the spec:\n" + "\n".join(f"  - {issue}" for issue in issues)


def _describe_yaml_error(exc: yaml.YAMLError) -> str:
    problem = getattr(exc, "problem", None)
    mark = getattr(exc, "problem_mark", None)
    if problem and mark is not None:
        return f"{problem} (column {mark.column + 1}). Check the indentation and quoting around that line."
    return str(exc)


def _validation_error_lines(exc: _PydanticValidationError, data: Any = None) -> list[str]:
    """One message per pydantic error (the CLI prints them as bullets)."""
    return [_render(None, None, issue.where, issue.message) for issue in _issues(exc, data, None)]


def _format_validation_error(exc: _PydanticValidationError, data: Any = None) -> str:
    return "; ".join(_validation_error_lines(exc, data))


def _issues(exc: _PydanticValidationError, data: Any, node: yaml.Node | None) -> list[_Issue]:
    issues = _merge_union_branches([_issue(error, data, node) for error in exc.errors()])
    # File order reads best; issues without a line keep pydantic's order, last.
    return sorted(issues, key=lambda issue: (issue.line is None, issue.line or 0))


def _issue(error: Any, data: Any, node: yaml.Node | None) -> _Issue:
    loc, branch, field_loc = _split_loc(tuple(error["loc"]))
    error_type = error["type"]
    message = error["msg"]
    got = _describe(error.get("input"))
    where_loc = loc

    if error_type == "extra_forbidden" and loc:
        key = loc[-1]
        where_loc = loc[:-1]
        allowed = _allowed_keys(where_loc)
        message = f"unknown key {key!r}.{did_you_mean(key, allowed)}"
        if allowed:
            message += f" Valid keys: {', '.join(allowed)}."
    elif error_type == "enum" and loc and loc[-1] in _ENUM_FIELDS:
        field = loc[-1]
        values = [member.value for member in _ENUM_FIELDS[field]]
        value = error.get("input")
        message = f"{value!r} is not a valid {field}.{did_you_mean(value, values)} Use {one_of(values)}."
    elif error_type == "missing" and loc:
        field = loc[-1]
        where_loc = loc[:-1]
        message = f"missing required key {field!r}."
        if field in _ENUM_FIELDS:
            message += f" Set it to {one_of(m.value for m in _ENUM_FIELDS[field])}."
    elif error_type in ("model_type", "dict_type"):
        message = f"expected a mapping of keys to values, got {got}."
        if loc and isinstance(loc[-1], int):
            message += " Each extraction point is a mapping with name, layers, tensor_type, position, ..."
    else:
        # pydantic prefixes custom ValueError messages raised inside our
        # validators with "Value error, "; strip that, our messages are
        # already self-describing.
        if message.startswith("Value error, "):
            message = message[len("Value error, ") :]
        if error_type not in ("value_error", "assertion_error"):
            message = f"{message} (got {got})"

    where = _where(where_loc, data)
    # The location already names the point; don't repeat it in the message.
    match = _POINT_PREFIX.match(message)
    if (
        match
        and len(where_loc) >= 2
        and where_loc[0] == "extraction_points"
        and _point_name(data, where_loc[1]) == match.group("name")
    ):
        message = message[match.end() :]

    return _Issue(
        where=where,
        message=message,
        line=_line(node, loc),
        field_loc=field_loc,
        branch=branch,
        error_type=error_type,
        got=got,
    )


def _split_loc(loc: tuple[Any, ...]) -> tuple[tuple[Any, ...], str | None, tuple[Any, ...]]:
    """Drop the union-branch tags pydantic puts in ``loc`` (``layers.int``,
    ``position.str``). No spec key is named like a tag.

    Returns (the cleaned loc, the branch tag or None, the loc before the tag).
    """
    kept: list[Any] = []
    branch: str | None = None
    field_loc: tuple[Any, ...] = ()
    for part in loc:
        if isinstance(part, str) and part in _UNION_BRANCHES:
            if branch is None:
                branch, field_loc = part, tuple(kept)
            continue
        kept.append(part)
    return tuple(kept), branch, field_loc


def _merge_union_branches(issues: list[_Issue]) -> list[_Issue]:
    """One issue per union-typed field instead of one per branch tried."""
    groups: dict[tuple[Any, ...], list[_Issue]] = {}
    order: list[_Issue | tuple[Any, ...]] = []
    for issue in issues:
        if issue.branch is None:
            order.append(issue)
        else:
            if issue.field_loc not in groups:
                groups[issue.field_loc] = []
                order.append(issue.field_loc)
            groups[issue.field_loc].append(issue)

    out: list[_Issue] = []
    for item in order:
        if isinstance(item, _Issue):
            out.append(item)
            continue
        group = groups[item]
        # A branch that got past the type check has the real complaint.
        specific = [i for i in group if not i.error_type.endswith("_type")]
        if specific:
            out.extend(specific)
            continue
        first = group[0]
        expected = " or ".join(_UNION_BRANCHES[i.branch] for i in group if i.branch)
        out.append(_Issue(where=first.where, message=f"expected {expected} (got {first.got})", line=first.line))
    return out


def _allowed_keys(loc: tuple[Any, ...]) -> list[str]:
    model: type[BaseModel] = ProbeSpecFile
    for part in loc:
        if isinstance(part, int):
            continue
        nested = _NESTED_MODELS.get((model, part))
        if nested is None:
            return []
        model = nested
    return list(model.model_fields)


def _point_name(data: Any, index: Any) -> str | None:
    if not isinstance(data, dict) or not isinstance(index, int):
        return None
    points = data.get("extraction_points")
    if not isinstance(points, list) or not 0 <= index < len(points):
        return None
    point = points[index]
    name = point.get("name") if isinstance(point, dict) else None
    return name if isinstance(name, str) and name else None


def _where(loc: tuple[Any, ...], data: Any) -> str:
    """``("extraction_points", 2, "position")`` -> ``extraction_points[2] 'drift'.position``."""
    out = ""
    for i, part in enumerate(loc):
        if isinstance(part, int):
            out += f"[{part}]"
            if i == 1 and loc[0] == "extraction_points":
                name = _point_name(data, part)
                if name is not None:
                    out += f" {name!r}"
        else:
            out += f".{part}" if out else str(part)
    return out


def _line(node: yaml.Node | None, loc: tuple[Any, ...]) -> int | None:
    """1-based line of the deepest YAML node ``loc`` reaches, or None."""
    if node is None:
        return None
    line = node.start_mark.line + 1
    for part in loc:
        if isinstance(node, yaml.MappingNode) and isinstance(part, str):
            spellings = (part, *_KEY_SPELLINGS.get(part, ()))
            for key_node, value_node in node.value:
                if isinstance(key_node, yaml.ScalarNode) and key_node.value in spellings:
                    line = key_node.start_mark.line + 1
                    node = value_node
                    break
            else:
                return line
        elif isinstance(node, yaml.SequenceNode) and isinstance(part, int) and 0 <= part < len(node.value):
            node = node.value[part]
            line = node.start_mark.line + 1
        else:
            return line
    return line


def _bad_position_index(data: Any) -> int | None:
    points = data.get("extraction_points") if isinstance(data, dict) else None
    if not isinstance(points, list):
        return None
    for index, point in enumerate(points):
        if isinstance(point, dict) and "position" in point:
            try:
                parse_position(point["position"])
            except PositionSyntaxError:
                return index
    return None


def _describe(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    return f"a {type(value).__name__}"

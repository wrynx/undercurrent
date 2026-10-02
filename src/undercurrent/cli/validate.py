"""``undercurrent validate``: check probe-spec files, with CI-friendly exit codes.

Exit 0 if every file is valid, 1 if any is invalid, 2 on a usage error.

Internal: not part of the public API (see docs/api-stability.md) and may
change in any release.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

from ._errors import UsageError


def register(subparsers: Any, parents: list[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "validate",
        parents=parents,
        help="check that probe-spec files are valid",
        description="Check that probe-spec YAML files are valid. Exits 0 if all are, 1 if any isn't, 2 on a usage error.",
    )
    parser.add_argument("specs", nargs="+", metavar="SPEC", help="spec file(s) to check")
    parser.add_argument(
        "--strict", action="store_true", help="treat deprecation warnings (e.g. old key names) as errors"
    )
    parser.add_argument(
        "--check-probes",
        action="store_true",
        help=(
            "also check that every probe_type is registered (built-in, installed plugins in the "
            "'undercurrent.probes' entry-point group, and modules given with --import)"
        ),
    )
    parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE",
        help="import MODULE first, so the probes it registers count for --check-probes (repeatable)",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text", help="output format (default: text)")
    parser.set_defaults(run=run)


@dataclass
class FileResult:
    path: str
    valid: bool
    extraction_points: int | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run(args: argparse.Namespace) -> int:
    if args.imports and not args.check_probes:
        raise UsageError("--import only makes sense with --check-probes; add --check-probes or drop --import")
    if args.check_probes:
        try:
            importlib.import_module("undercurrent.core.registry")
        except ImportError as exc:
            raise UsageError(f"--check-probes needs the probe registry, which failed to import: {exc}") from exc
    for module in args.imports:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise UsageError(
                f"--import {module}: {exc}. Check the module name (dotted, e.g. my_package.probes) and that its "
                "package is installed in this environment."
            ) from exc

    results = [validate_file(path, strict=args.strict, check_probes=args.check_probes) for path in args.specs]

    if args.format == "json":
        json.dump([asdict(r) for r in results], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for result in results:
            _print_text(result)
    return 0 if all(r.valid for r in results) else 1


def validate_file(path: str, *, strict: bool = False, check_probes: bool = False) -> FileResult:
    """Validate one spec file. Never raises for a bad file; the problems go in the result."""
    from ..spec import ProbingSpecError, parse_yaml

    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as exc:
        return FileResult(path, valid=False, errors=[_describe_read_error(exc)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            spec = parse_yaml(text)
        except ProbingSpecError as exc:
            return FileResult(path, valid=False, errors=_issues(exc))
    deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]

    result = FileResult(path, valid=True, extraction_points=len(spec.extraction_points))
    if strict:
        result.errors += deprecations
    else:
        result.warnings += deprecations
    if check_probes:
        result.errors += _probe_issues(spec)
    result.valid = not result.errors
    return result


def _issues(exc: Exception) -> list[str]:
    """One message per problem in a spec parse error (``line N: where: problem``)."""
    return list(getattr(exc, "issues", None) or (str(exc),))


def _probe_issues(spec: Any) -> list[str]:
    from ..core.registry import ProbeNotFoundError, get_probe_factory

    issues = []
    missing: dict[str, list[str]] = {}
    for point in spec.extraction_points:
        missing.setdefault(point.probe_type, []).append(point.name)
    for probe_type, points in missing.items():
        try:
            get_probe_factory(probe_type)
        except ProbeNotFoundError as exc:
            where = ", ".join(repr(p) for p in points)
            issues.append(f"extraction point(s) {where}: {exc}")
    return issues


def _describe_read_error(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "file not found; check the path (relative paths are resolved against the current directory)"
    if isinstance(exc, IsADirectoryError):
        return "is a directory, not a spec file"
    if isinstance(exc, UnicodeDecodeError):
        return f"not a UTF-8 text file ({exc.reason})"
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return str(exc)


def _print_text(result: FileResult) -> None:
    if result.valid:
        n = result.extraction_points
        print(f"{result.path}: OK ({n} extraction point{'' if n == 1 else 's'})")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        return
    n = len(result.errors)
    print(f"{result.path}: ERROR {n} issue{'' if n == 1 else 's'}")
    for issue in result.errors:
        print(f"  - {issue}")
    for warning in result.warnings:
        print(f"  warning: {warning}")

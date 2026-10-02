"""The ``undercurrent`` command-line tool.

Three subcommands, each in its own module with ``register(subparsers)`` and
``run(args) -> int``::

    undercurrent inspect-model MODEL     # what can I probe in this model?
    undercurrent validate SPEC [SPEC...] # is my spec valid? (CI-friendly exit codes)
    undercurrent schema [-o FILE]        # the spec JSON Schema

Subcommand modules must not import torch, transformers or vllm at import
time, so ``undercurrent --help`` stays fast; they import them inside
``run()``.

Exit codes: 0 success, 1 a failure the command reports (an invalid spec, a
model that can't be inspected), 2 a usage error.

The public Python surface is `main()` (the console-script entry point).
The command-line interface itself -- commands, options, exit codes -- is the
stable contract; the subcommand modules, `build_parser` and the CLI error
classes are internal.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from ..errors import ProbingError
from ._errors import UsageError

#: Subcommand modules, in help order. The command name is the module name
#: with dashes (``inspect_model`` -> ``inspect-model``).
_COMMANDS = ("inspect_model", "validate", "schema")


def _version() -> str:
    from .. import __version__

    return __version__


def build_parser() -> argparse.ArgumentParser:
    """The top-level ``undercurrent`` parser, with every subcommand registered."""
    parser = argparse.ArgumentParser(
        prog="undercurrent",
        description="Undercurrent: Wrynx's activation-probing platform. See the undercurrent before it surfaces.",
    )
    parser.add_argument("--version", action="version", version=f"undercurrent {_version()}")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks for errors")

    # Lets --debug also follow the subcommand; SUPPRESS keeps it from
    # resetting a --debug given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(title="commands", metavar="COMMAND")
    for name in _COMMANDS:
        module = importlib.import_module(f".{name}", __name__)
        module.register(subparsers, parents=[common])
    return parser


def _is_user_error(exc: BaseException) -> bool:
    """Errors that mean "the user asked for something that can't be done"
    rather than "undercurrent has a bug": printed as ``error: ...`` without a
    traceback (unless --debug). Every deliberate library error is a
    `ProbingError` (`CLIError` included)."""
    return isinstance(exc, (ProbingError, OSError))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``undercurrent`` CLI and return its exit code.

    This is the console-script entry point. It doesn't call ``sys.exit``, so
    it can be called from Python:

    ```python
    from undercurrent.cli import main

    code = main(["validate", "probes.yaml"])
    ```

    Args:
        argv: the arguments, without the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The exit code: 0 success, 1 a reported failure, 2 a usage error,
            130 interrupted.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version and argparse usage errors
        return exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 2

    run = getattr(args, "run", None)
    if run is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if args.debug or not _is_user_error(exc):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, UsageError) else 1


__all__ = ["main"]

"""``undercurrent schema``: print the probe-spec JSON Schema, or write it to a file.

Internal: not part of the public API (see docs/api-stability.md) and may
change in any release.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def register(subparsers: Any, parents: list[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "schema",
        parents=parents,
        help="print the probe-spec JSON Schema",
        description=(
            "Print the JSON Schema (Draft 2020-12) for probe-spec YAML files, for editor "
            "autocompletion and for validating specs with other tools."
        ),
    )
    parser.add_argument("-o", "--output", metavar="FILE", help="write the schema to FILE instead of stdout")
    parser.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from ..spec import json_schema

    # Same rendering as the committed schema/probe-spec.schema.json.
    text = json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0

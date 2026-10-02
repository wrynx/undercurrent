#!/usr/bin/env python
"""Write the probe-spec JSON Schema to schema/probe-spec.schema.json.

Run this after changing the spec models (src/undercurrent/spec/schema.py) or
src/undercurrent/spec/json_schema.py, and commit the result:

    python scripts/generate_schema.py

tests/spec/test_json_schema.py fails while the committed file is stale.
"""

from __future__ import annotations

import json
from pathlib import Path

from undercurrent.spec import json_schema

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "probe-spec.schema.json"


def render() -> str:
    """The exact file contents: 2-space indented JSON plus a trailing newline."""
    return json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()

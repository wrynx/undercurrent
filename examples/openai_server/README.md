# OpenAI-compatible wire format for probe verdicts (reference example)

`response_schema.py` is a reference implementation of an OpenAI-compatible wire format for probe verdicts: completion bodies, `probing` objects and SSE chunks.
It is **not part of the installed `undercurrent` package**; copy or import it from your own serving layer (it depends only on `undercurrent.spec`).
Tests live in `tests/examples/openai_server/`. A built-in `undercurrent serve` server is on the roadmap.

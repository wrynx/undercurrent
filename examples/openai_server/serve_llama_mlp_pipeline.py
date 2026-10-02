#!/usr/bin/env python
"""Reference HTTP server: `CustomMLPProbe` attached to a live Llama-3.1-8B
model via vLLM, serving concurrent `/generate` requests.

Stack: undercurrent.spec (spec) -> undercurrent.router (Router) -> undercurrent.adapters.vllm
(VLLMEngineAdapter) -> content_safety_demo.CustomMLPProbe (a REAL trained
MLP checkpoint, not the structural random-init probes elsewhere in the
content-safety demo -- see examples/content_safety/content_safety_demo/custom_mlp.py).

Model access: the default `--model`, `meta-llama/Llama-3.1-8B`, is a GATED
Hugging Face repo -- accept its license on the model page and run
`huggingface-cli login` before first use. For an ungated first run, start
with `examples/content_safety/run_vllm_pipeline.py` (defaults to `gpt2`) instead. Pointing
`--model` at a different model also means a matching `--spec` (layer
indices) and a checkpoint sized to that model's hidden_size.

Concurrency model: `http.server.ThreadingHTTPServer` spawns one thread per
in-flight HTTP request; each request thread calls `adapter.generate(...)`
independently. `VLLMEngineAdapter.generate()` is safe to call concurrently
from multiple threads -- vLLM's own continuous batching actually shares
scheduler steps across them (this is the scenario
`tests/adapters/vllm/test_integration_vllm.py`'s
`test_two_concurrent_requests_share_batching_and_route_correctly` proves at
the unit level; this script is that same pattern, wired into a real server).
`Router` and `VLLMEngineAdapter` are each constructed ONCE at startup and
shared across every request -- per-request state (probe instances, the
verdicts a request's probes produced) is always keyed by that request's own
`request_id`, never shared across requests (see `undercurrent.router`'s own
isolation-model tests for why concurrent requests don't see each other's
state).

`http.server.ThreadingHTTPServer` is stdlib-only (no new dependency), but
isn't hardened for production traffic (no backpressure, no request
timeouts, no TLS) -- swap in a real ASGI/WSGI server (uvicorn, gunicorn,
...) in front of the same `run_generate()` request-handling logic below for
production use; nothing about the pipeline wiring itself needs to change.

Setup
-----
    export PROBE_MODEL_PATH=/path/to/checkpoint.pt   # see custom_mlp.py's checkpoint format
    # No real checkpoint yet? Generate a placeholder (random weights, sized
    # for Llama-3.1-8B's hidden_size=4096):
    python examples/content_safety/make_dummy_mlp_checkpoint.py --input-dim 4096 --out "$PROBE_MODEL_PATH"

    # from the repo root; the demo itself is not installed
    pip install -e .            # undercurrent, incl. undercurrent.adapters.vllm
    pip install -e '.[vllm]'    # or install into an env that already has a GPU/CUDA-matched vllm+torch
    huggingface-cli login   # meta-llama/Llama-3.1-8B is gated -- accept its license on HF first

Run
---
    python examples/openai_server/serve_llama_mlp_pipeline.py --port 8000

The server listens on 127.0.0.1 by default, so only the local machine can
reach it. In a container (or to accept connections from other hosts on a
trusted network), pass `--host 0.0.0.0`; the example Dockerfile does. There is
no authentication, so don't expose it to an untrusted network.

Use (from another shell -- run several at once to see concurrent serving):
    curl -s localhost:8000/generate -d '{"prompt": "Tell me about the ocean.", "max_tokens": 64}' &
    curl -s localhost:8000/generate -d '{"prompt": "Write a haiku about rain.", "max_tokens": 32}' &
    wait
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from undercurrent import ProbeFactory, ProbeResult, Router
from undercurrent.spec import ProbeSpec, load_yaml_file

# The demo probes and their YAML specs live in the sibling examples/content_safety/ (not installed).
_CONTENT_SAFETY_DIR = Path(__file__).resolve().parents[1] / "content_safety"
sys.path.insert(0, str(_CONTENT_SAFETY_DIR))

from content_safety_demo.spec_binding import probe_kwargs_from_extraction_point, resolve_probe_cls

logger = logging.getLogger("serve_llama_mlp_pipeline")

DEFAULT_SPEC_PATH = _CONTENT_SAFETY_DIR / "custom_mlp_llama31_8b.yaml"

_verdicts_lock = threading.Lock()
_pending_verdicts: dict[str, dict[str, ProbeResult]] = {}


def _capture_verdicts(request_id: str, results: dict[str, ProbeResult]) -> None:
    """`Router.on_request_end` listener: stash each request's results in
    `_pending_verdicts`, keyed by request_id. `adapter.generate()` calls
    `router.end_request()` itself and discards the results, so this is how
    `run_generate()` gets them back. Keying by request_id (not a single
    shared slot) is required once multiple requests are genuinely
    concurrent.
    """
    with _verdicts_lock:
        _pending_verdicts[request_id] = results


def build_probe_registry(
    spec: ProbeSpec,
    *,
    threshold: float,
    model_path: str,
    device: str,
    activation: str,
    final_activation: bool,
) -> dict[str, Any]:
    """Build once at server startup, shared across every request. Passes
    `model_path`/`device`/`activation`/`final_activation` as overrides to
    every extraction point's probe -- fine for this script's spec (a single
    `custom_mlp_safety_check` point), but note this would need per-probe-type
    override dispatch if you extend the spec with a probe type that doesn't
    accept those kwargs.
    """
    return {
        point.probe_type: ProbeFactory(
            resolve_probe_cls(point),
            probe_kwargs_from_extraction_point(
                point,
                threshold=threshold,
                model_path=model_path,
                device=device,
                activation=activation,
                final_activation=final_activation,
            ),
        )
        for point in spec
    }


def run_generate(
    adapter, router: Router, spec: ProbeSpec, prompt: str, max_tokens: int, temperature: float
) -> dict[str, Any]:
    """One full request: register -> generate -> collect this request's
    verdicts -> unregister. Safe to call concurrently from multiple threads
    for different requests -- see the module docstring."""
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    adapter.register_extraction(request_id, list(spec))
    try:
        text = adapter.generate(request_id, prompt, {"max_tokens": max_tokens, "temperature": temperature}, router)
    finally:
        adapter.unregister_extraction(request_id)

    with _verdicts_lock:
        results = _pending_verdicts.pop(request_id, {})

    return {
        "request_id": request_id,
        "prompt": prompt,
        "text": text,
        "verdicts": {name: result.verdict for name, result in results.items()},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "ProbingMLPPipeline/0.1"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming convention
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok", "model": self.server.model_name})  # type: ignore[attr-defined]
            return
        self._send_json(404, {"error": f"unknown path {self.path!r}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/generate":
            self._send_json(404, {"error": f"unknown path {self.path!r}"})
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            self._send_json(400, {"error": "'prompt' (non-empty string) is required"})
            return
        max_tokens = int(body.get("max_tokens", self.server.default_max_tokens))  # type: ignore[attr-defined]
        temperature = float(body.get("temperature", 0.0))

        try:
            result = run_generate(
                self.server.adapter,  # type: ignore[attr-defined]
                self.server.router,  # type: ignore[attr-defined]
                self.server.spec,  # type: ignore[attr-defined]
                prompt,
                max_tokens,
                temperature,
            )
        except Exception as exc:  # noqa: BLE001 -- reference server: report the error, don't crash the process
            logger.exception("/generate failed")
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, result)

    def log_message(self, fmt: str, *args: Any) -> None:  # override to route through logging, not stderr directly
        logger.info("%s - %s", self.address_string(), fmt % args)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B",
        help="HF model name or local path (the default is a gated repo: accept its license on Hugging Face "
        "and run `huggingface-cli login` first)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to listen on (default: 127.0.0.1, local only; use 0.0.0.0 in a container)",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC_PATH), help="path to the ExtractionPoint YAML spec")
    parser.add_argument("--threshold", type=float, default=0.5, help="flag threshold in [0, 1]")
    parser.add_argument(
        "--device", default="cpu", help="device the MLP probe head runs on (independent of vLLM's own device)"
    )
    parser.add_argument(
        "--activation",
        default="relu",
        choices=["relu", "gelu"],
        help="hidden-layer nonlinearity the checkpoint was trained with (can't be inferred from the checkpoint itself)",
    )
    parser.add_argument(
        "--final-activation",
        action="store_true",
        help="apply --activation to the output logits before sigmoid/softmax too, matching training code that "
        "runs the same activation module across every layer of the head, output layer included",
    )
    parser.add_argument("--default-max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--allow-unsupported-executor", action="store_true")
    parser.add_argument("--allow-v2-model-runner", action="store_true")
    parser.add_argument("--allow-request-id-randomization", action="store_true")
    args = parser.parse_args()

    model_path = os.environ.get("PROBE_MODEL_PATH")
    if not model_path:
        print(
            "PROBE_MODEL_PATH env var is required -- point it at a CustomMLPProbe checkpoint. "
            "No real one yet? Generate a placeholder:\n"
            "  python examples/content_safety/make_dummy_mlp_checkpoint.py --input-dim 4096 --out /tmp/mlp_probe.pt\n"
            "  export PROBE_MODEL_PATH=/tmp/mlp_probe.pt",
            file=sys.stderr,
        )
        return 1
    if not os.path.exists(model_path):
        print(f"PROBE_MODEL_PATH={model_path!r} does not exist.", file=sys.stderr)
        return 1

    try:
        from undercurrent.adapters.vllm import VLLMEngineAdapter
    except ImportError:
        print(
            "undercurrent.adapters.vllm is not importable. Install undercurrent into an "
            "environment with a GPU-matched vllm/torch build first:\n"
            "  pip install -e ..            # into your existing vLLM environment, or\n"
            "  pip install -e '..[vllm]'    # pulls in vLLM via the extra\n"
            "See the vLLM adapter's legacy guide in docs/_legacy/ for details.",
            file=sys.stderr,
        )
        return 1

    spec = load_yaml_file(args.spec)
    probe_registry = build_probe_registry(
        spec,
        threshold=args.threshold,
        model_path=model_path,
        device=args.device,
        activation=args.activation,
        final_activation=args.final_activation,
    )
    router = Router(probe_registry=probe_registry)
    router.on_request_end(_capture_verdicts)

    adapter = VLLMEngineAdapter()
    logger.info("loading %r via vLLM ...", args.model)
    adapter.load_model(
        args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        allow_unsupported_executor=args.allow_unsupported_executor,
        allow_v2_model_runner=args.allow_v2_model_runner,
        allow_request_id_randomization=args.allow_request_id_randomization,
    )

    logger.info("warming up (forces router binding + kernel JIT compilation before accepting real traffic) ...")
    try:
        run_generate(adapter, router, spec, "Hello, world.", max_tokens=1, temperature=0.0)
    except Exception:
        logger.exception("warmup request failed")
        adapter.shutdown()
        return 1

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.adapter = adapter  # type: ignore[attr-defined]
    httpd.router = router  # type: ignore[attr-defined]
    httpd.spec = spec  # type: ignore[attr-defined]
    httpd.model_name = args.model  # type: ignore[attr-defined]
    httpd.default_max_tokens = args.default_max_tokens  # type: ignore[attr-defined]

    logger.info("serving on http://%s:%s (POST /generate, GET /healthz) -- Ctrl+C to stop", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutting down ...")
        httpd.shutdown()
        adapter.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

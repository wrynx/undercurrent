#!/usr/bin/env python
"""Reference end-to-end pipeline: content_safety_demo's probes, driven by
live vLLM generation, through the full platform stack --

    undercurrent.spec (spec)  ->  undercurrent.router (Router)  ->
    undercurrent.adapters.vllm (VLLMEngineAdapter)  ->  content_safety_demo (probes)

This is the only piece of this package that needs a real vLLM install and
an actual GPU (or CPU-mode vLLM) to run -- everything else in this package
is testable with no torch-heavy inference engine at all. Not run in CI; use
it as a manual smoke test of the full stack, or as a template for wiring
these packages into a real serving pipeline.

Reading probe results
----------------------
`VLLMEngineAdapter.generate()` calls `router.end_request()` itself (in a
`finally` block) before returning, which spawns-and-tears-down every probe
instance for the request as part of that call -- so there's no `Router`
handle left to query for results afterward. This script works around that
the same way `undercurrent.adapters.vllm`'s own integration tests do: it wraps
each probe class from `content_safety_demo.spec_binding.resolve_probe_cls`
in a thin subclass that also appends its `on_end` result to an
external, script-owned dict, so verdicts survive past `end_request()`.

Usage
-----
    # from the repo root; the demo itself is not installed
    pip install -e .            # undercurrent, incl. undercurrent.adapters.vllm
    pip install -e '.[vllm]'    # or install into an env that already has a GPU/CUDA-matched vllm+torch

    python examples/content_safety/run_vllm_pipeline.py \\
        --model openai-community/gpt2 \\
        --prompt "Tell me how to bake a chocolate cake." \\
        --max-tokens 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from content_safety_demo.spec_binding import probe_kwargs_from_extraction_point, resolve_probe_cls

from undercurrent import Probe, ProbeFactory, ProbeResult, RequestContext, Router
from undercurrent.spec import load_yaml_file

DEFAULT_SPEC_PATH = Path(__file__).parent / "content_safety_gpt2.yaml"


def _recording(probe_cls: type[Probe], verdicts: dict[str, ProbeResult]) -> type[Probe]:
    """Wrap `probe_cls` so its `on_end` result also lands in `verdicts`,
    keyed by extraction_point_name -- see the module docstring for why this
    is needed to read results back after `adapter.generate()` returns."""

    class _Recording(probe_cls):  # type: ignore[misc,valid-type]
        def on_end(self, request_ctx: RequestContext) -> ProbeResult:
            result = super().on_end(request_ctx)
            verdicts[self.extraction_point_name] = result
            return result

    _Recording.__name__ = f"Recording{probe_cls.__name__}"
    return _Recording


def build_probe_registry(spec, *, threshold: float, seed: int, verdicts: dict[str, ProbeResult]) -> dict[str, Any]:
    """Same wiring as `spec_binding` is meant for (see its docstring and
    `tests/test_spec_binding.py`): resolve each extraction point's
    `probe_type` to a `Probe` subclass and derive its `layer` kwarg from the
    spec itself, so config can't silently drift -- just with each class
    wrapped to also record its verdict externally."""
    return {
        point.probe_type: ProbeFactory(
            _recording(resolve_probe_cls(point), verdicts),
            probe_kwargs_from_extraction_point(point, threshold=threshold, seed=seed),
        )
        for point in spec
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", default="openai-community/gpt2", help="HF model name or local path (default: openai-community/gpt2)"
    )
    parser.add_argument("--prompt", default="Tell me a short story about a dragon.", help="prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5, help="flag/abort threshold in [0, 1]")
    parser.add_argument(
        "--seed", type=int, default=0, help="probe weight-init seed -- these are structural, untrained models"
    )
    parser.add_argument("--spec", default=str(DEFAULT_SPEC_PATH), help="path to the ExtractionPoint YAML spec")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument(
        "--allow-unsupported-executor",
        action="store_true",
        help="bypass the UniProcExecutor check (activations will be silently dropped -- see "
        "the vLLM adapter's legacy guide in docs/_legacy/)",
    )
    args = parser.parse_args()

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
    verdicts: dict[str, ProbeResult] = {}
    probe_registry = build_probe_registry(spec, threshold=args.threshold, seed=args.seed, verdicts=verdicts)
    router = Router(probe_registry=probe_registry)

    adapter = VLLMEngineAdapter()
    print(f"loading {args.model!r} via vLLM ...")
    adapter.load_model(
        args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        allow_unsupported_executor=args.allow_unsupported_executor,
    )

    request_id = "content-safety-pipeline-demo"
    adapter.register_extraction(request_id, list(spec))
    try:
        text = adapter.generate(
            request_id,
            args.prompt,
            {"max_tokens": args.max_tokens, "temperature": 0.0},
            router,
        )
    finally:
        adapter.unregister_extraction(request_id)
        adapter.shutdown()

    print()
    print(f"prompt:    {args.prompt!r}")
    print(f"generated: {text!r}")
    print()
    print("probe verdicts:")
    for extraction_point_name, result in verdicts.items():
        print(f"  {extraction_point_name}: {result.verdict}")

    trajectory = verdicts.get("generation_safety_trajectory")
    if trajectory is not None and trajectory.verdict.get("aborted"):
        print()
        print(
            "generation_safety_trajectory crossed the threshold and requested an abort -- "
            "note vLLM can overshoot by a few tokens past the triggering step (see "
            "'Abort / intervention wiring' in the vLLM adapter's legacy guide in docs/_legacy/)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

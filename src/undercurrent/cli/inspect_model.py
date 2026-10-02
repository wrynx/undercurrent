"""``undercurrent inspect-model``: what can I probe in this model?

Reads only the model's ``config.json`` (a local directory works fully
offline), instantiates the model on the ``meta`` device (no weights are
downloaded or allocated) and reports where each adapter would hook it. If
the model can't be instantiated that way, it falls back to the numbers in
the config and marks the support table ``unverified``.

Internal: not part of the public API (see docs/api-stability.md) and may
change in any release.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ._errors import CLIError

BACKENDS = ("hf", "vllm")
SUGGESTED_POSITION = "prompt[-1]"
SUGGESTED_PROBE_TYPE = "my_probe"


def register(subparsers: Any, parents: list[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "inspect-model",
        parents=parents,
        help="list a model's hookable layers and tensor types",
        description=(
            "List a model's decoder layers and which tensor types each backend can capture. Only "
            "config.json is fetched; the model is built on the meta device, so no weights are "
            "downloaded or allocated."
        ),
    )
    parser.add_argument("model", metavar="MODEL", help="Hugging Face model id or local model directory")
    parser.add_argument(
        "--backend",
        type=_backend_choice,
        default="all",
        metavar="{" + ",".join((*BACKENDS, "all")) + "}",
        help="which adapter(s) to report on (default: all)",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text", help="output format (default: text)")
    parser.add_argument(
        "--emit-spec", metavar="FILE", help="also write a commented starter spec for this model to FILE"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow custom model code from the Hub (only for repos you trust)",
    )
    parser.add_argument("--revision", metavar="REV", help="model revision (branch, tag or commit) to read")
    parser.set_defaults(run=run)


def _backend_choice(value: str) -> str:
    """argparse ``type=`` for ``--backend``: like ``choices=``, plus a did-you-mean."""
    from ..errors import did_you_mean, one_of

    choices = (*BACKENDS, "all")
    if value not in choices:
        raise argparse.ArgumentTypeError(
            f"unknown backend {value!r}.{did_you_mean(value, choices)} Choose {one_of(choices)}."
        )
    return value


def run(args: argparse.Namespace) -> int:
    backends = BACKENDS if args.backend == "all" else (args.backend,)
    report = inspect_model(
        args.model, backends=backends, trust_remote_code=args.trust_remote_code, revision=args.revision
    )
    if args.emit_spec:
        text = starter_spec(report)
        with open(args.emit_spec, "w", encoding="utf-8") as f:
            f.write(text)

    if args.format == "json":
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_text(report)
    if args.emit_spec:
        print(
            f"wrote starter spec {args.emit_spec}; check it with: undercurrent validate {args.emit_spec}",
            file=sys.stderr,
        )
    return 0


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def inspect_model(
    model: str,
    *,
    backends: tuple[str, ...] = BACKENDS,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable report ``inspect-model`` prints."""
    from ..adapters._optional import require_transformers

    transformers = require_transformers()
    try:
        config = transformers.AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code, revision=revision)
    except ValueError as exc:  # unrecognized model_type, remote code not trusted, ...
        raise CLIError(
            f"could not load the config for {model!r}: {exc}. Check the model id or path; for a model with "
            "custom code from a repo you trust, add --trust-remote-code."
        ) from exc
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config

    meta_model, note = _instantiate_on_meta(config, trust_remote_code=trust_remote_code)
    verified = meta_model is not None

    if verified:
        from ..adapters.hf.introspect import find_decoder_layers

        # Raises HFAdapterLimitationError (a helpful, user-facing error) for an
        # architecture neither adapter knows how to hook; the vLLM worker
        # looks for decoder layers at the same attribute paths.
        num_layers: int | None = len(find_decoder_layers(meta_model))
        model_class = type(meta_model).__name__
    else:
        num_layers = getattr(text_config, "num_hidden_layers", None)
        architectures = getattr(config, "architectures", None) or [None]
        model_class = architectures[0]

    support = {backend: _backend_support(backend, meta_model) for backend in backends}
    suggested_layer = num_layers // 2 if num_layers else None
    return {
        "model": model,
        "model_class": model_class,
        "model_type": getattr(config, "model_type", None),
        "num_layers": num_layers,
        "hidden_size": getattr(text_config, "hidden_size", None),
        "num_attention_heads": getattr(text_config, "num_attention_heads", None),
        "vocab_size": getattr(text_config, "vocab_size", None),
        "weights_loaded": False,
        "verified": verified,
        "note": note,
        "support": support,
        "suggested": None
        if suggested_layer is None
        else {"layers": [suggested_layer], "tensor_type": "residual_stream", "position": SUGGESTED_POSITION},
    }


def _instantiate_on_meta(config: Any, *, trust_remote_code: bool) -> tuple[Any, str | None]:
    """Build the model from its config on the meta device: real module
    structure, no weights. Returns ``(model, None)``, or ``(None, note)`` if
    that isn't possible."""
    from ..adapters._optional import require_torch, require_transformers

    torch = require_torch()
    transformers = require_transformers()
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        with torch.device("meta"):
            model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - any failure here just means "fall back to config numbers"
        return None, (
            f"could not build the model on the meta device ({type(exc).__name__}: {exc}); the numbers "
            "come from config.json and tensor-type support is unverified"
        )
    finally:
        transformers.logging.set_verbosity(verbosity)
    return model, None


def _backend_support(backend: str, meta_model: Any) -> dict[str, dict[str, str | None]]:
    if backend == "hf":
        from ..adapters.hf import introspect as hf_support

        backend_support: Any = hf_support
    else:
        from ..adapters.vllm import support as vllm_support

        backend_support = vllm_support

    if meta_model is None:
        table = backend_support.supported_tensor_types()
        return {
            t.value: {"status": "no", "reason": reason} if reason else {"status": "unverified", "reason": None}
            for t, reason in table.items()
        }
    table = backend_support.model_tensor_support(meta_model)
    return {
        t.value: {"status": "no", "reason": reason} if reason else {"status": "yes", "reason": None}
        for t, reason in table.items()
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_text(report: dict[str, Any]) -> None:
    def show(value: Any) -> str:
        return "unknown" if value is None else str(value)

    print(f"model:       {report['model']}")
    print(f"class:       {show(report['model_class'])} (model_type={show(report['model_type'])})")
    print(f"layers:      {show(report['num_layers'])}")
    print(f"hidden size: {show(report['hidden_size'])}")
    print(f"heads:       {show(report['num_attention_heads'])}")
    print(f"vocab:       {show(report['vocab_size'])}")
    if report["verified"]:
        print("weights:     not loaded (structure built on the meta device)")
    else:
        print(f"note:        {report['note']}")
    print()

    backends = list(report["support"])
    reasons: list[str] = []
    rows = []
    for tensor_type in report["support"][backends[0]]:
        cells = []
        for backend in backends:
            cell = report["support"][backend][tensor_type]
            if cell["status"] == "no":
                if cell["reason"] not in reasons:
                    reasons.append(cell["reason"])
                cells.append(f"no [{reasons.index(cell['reason']) + 1}]")
            else:
                cells.append(cell["status"])
        rows.append([tensor_type, *cells])
    header = ["tensor_type", *backends]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    for i, reason in enumerate(reasons, 1):
        print(f"  [{i}] {reason}")
    if "vllm" in backends:
        print(
            "  vllm: checked against this model's transformers module names, which vLLM's own model "
            "implementations follow for the common families; not checked against vLLM itself."
        )
        if report["support"]["vllm"]["final_norm"]["status"] != "no" and report["num_layers"]:
            print(f"  vllm: final_norm is captured at layer index {report['num_layers']} (one past the last layer).")

    suggested = report["suggested"]
    if suggested:
        print()
        print(
            f"Probe it with layers 0..{report['num_layers'] - 1}, e.g. layers: {suggested['layers']}, "
            f'position: "{suggested["position"]}"'
        )


_SPEC_TEMPLATE = """\
{header}
# Starter probe spec for {model} ({num_layers} decoder layers: 0..{last_layer}),
# written by `undercurrent inspect-model --emit-spec`.
# Check it with: undercurrent validate <this file>
version: "1"
extraction_points:
  # Capture the residual stream (the decoder layer's output) at the middle
  # layer, at the last prompt token, and run a single-shot probe on it
  # inline (on the generation path, so its result can stop generation).
  - name: middle_layer_last_prompt_token
    layers: [{layer}]
    tensor_type: residual_stream
    position: "{position}"
    # The name your probe is registered under, e.g. @register_probe("{probe_type}").
    probe_type: {probe_type}
    probe_kind: single_shot
    execution_mode: inline
"""


def starter_spec(report: dict[str, Any]) -> str:
    """A commented starter spec (YAML) for the inspected model, checked by parsing it back."""
    from ..spec import ProbingSpecError, TensorType, parse_yaml
    from ..spec.json_schema import SCHEMA_ID

    suggested = report["suggested"]
    if suggested is None:
        raise CLIError(
            f"cannot write a starter spec for {report['model']!r}: its layer count is unknown (the config has "
            "no num_hidden_layers). Write the spec by hand; `undercurrent schema` prints the format."
        )
    layer = suggested["layers"][0]
    model_name = " ".join(str(report["model"]).split())  # keep the comment on one line
    text = _SPEC_TEMPLATE.format(
        header=f"# yaml-language-server: $schema={SCHEMA_ID}",
        model=model_name,
        num_layers=report["num_layers"],
        last_layer=report["num_layers"] - 1,
        layer=layer,
        position=suggested["position"],
        probe_type=SUGGESTED_PROBE_TYPE,
    )
    try:
        spec = parse_yaml(text)
    except ProbingSpecError as exc:  # pragma: no cover - the template is static
        raise RuntimeError(f"the starter spec template is invalid: {exc}") from exc
    (point,) = spec.extraction_points
    assert point.layers == (layer,) and point.tensor_type == TensorType.RESIDUAL_STREAM
    return text

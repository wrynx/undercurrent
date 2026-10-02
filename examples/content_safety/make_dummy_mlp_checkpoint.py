#!/usr/bin/env python
"""Write a placeholder `CustomMLPProbe` checkpoint to disk.

There's no real trained content-safety classifier in this repo -- this
script produces a structurally correct but RANDOMLY INITIALIZED
`SafetyClassifierHead` checkpoint (see `content_safety_demo.models`),
sized to match a real model's hidden dimension, purely so
`examples/openai_server/serve_llama_mlp_pipeline.py` has something loadable to point
`PROBE_MODEL_PATH` at. Swap it for a real checkpoint (same format --
`torch.save(model.state_dict(), path)` for a `SafetyClassifierHead`) once
you have one; nothing downstream needs to change.

Usage:
    python examples/content_safety/make_dummy_mlp_checkpoint.py --input-dim 4096 --out /tmp/mlp_probe.pt
    # or, to match wherever the server will look:
    PROBE_MODEL_PATH=/tmp/mlp_probe.pt python examples/content_safety/make_dummy_mlp_checkpoint.py --input-dim 4096
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from content_safety_demo.models import SafetyClassifierHead


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dim",
        type=int,
        default=4096,
        help="activation dimension the checkpoint expects (default: 4096, Llama-3.1-8B's hidden_size)",
    )
    parser.add_argument("--hidden-size", type=int, default=128, help="classifier head's hidden layer size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=os.environ.get("PROBE_MODEL_PATH"),
        help="output path (default: $PROBE_MODEL_PATH)",
    )
    args = parser.parse_args()

    if not args.out:
        print(
            "No output path given and PROBE_MODEL_PATH isn't set. Pass --out /path/to/checkpoint.pt "
            "or set PROBE_MODEL_PATH first.",
            file=sys.stderr,
        )
        return 1

    torch.manual_seed(args.seed)
    model = SafetyClassifierHead(args.input_dim, args.hidden_size)
    torch.save(model.state_dict(), args.out)
    print(
        f"wrote a random-initialized SafetyClassifierHead(input_dim={args.input_dim}, "
        f"hidden_size={args.hidden_size}) checkpoint to {args.out!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train a linear (or small MLP) probe on a model's activations with Undercurrent.

    python examples/train_probe/train_probe.py --model openai-community/gpt2 --dataset toy --layer 6 --epochs 20

Pipeline: load a labelled dataset (`data_sources.py`), collect one
activation per prompt with `ProbedModel` + `ActivationCollectorProbe`
(`max_new_tokens=1`), split train/val/test (stratified 70/15/15), train with
plain torch, report accuracy, precision, recall, F1 and AUROC, then write
`probe.safetensors` + `probe.json` (see `linear_probe.py`) and `metrics.json`
to `--out`. Run the result with `use_probe.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from collector import ActivationCollectorProbe
from data_sources import SOURCES, load_examples
from linear_probe import ARCHITECTURES, ProbeHead, save_probe

from undercurrent import ProbedModel, ProbeFactory

COLLECT_POINT = "collect"
COLLECTOR_TYPE = "activation_collector"
TENSOR_TYPE = "residual_stream"
SPLITS = (0.70, 0.15, 0.15)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        default="openai-community/gpt2",
        help="Hugging Face model id or local path (default: openai-community/gpt2)",
    )
    parser.add_argument("--dataset", default="toy", choices=sorted(SOURCES), help="data source (default: toy)")
    parser.add_argument("--layer", type=int, required=True, help="layer index to probe (0-based)")
    parser.add_argument("--position", default="prompt[-1]", help="spec position to read (default: prompt[-1])")
    parser.add_argument("--limit", type=int, default=None, help="number of examples (class-balanced)")
    parser.add_argument("--probe", default="linear", choices=ARCHITECTURES, help="probe architecture")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden size (default: 64)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5, help="score at which the probe flags (default: 0.5)")
    parser.add_argument("--out", default="outputs/train_probe", help="output directory (default: outputs/train_probe)")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:1, ... (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    return args


# --- collection -----------------------------------------------------------------


def collect_activations(
    model: str, texts: Sequence[str], *, layer: int, position: str, device: str
) -> tuple[torch.Tensor, list[int]]:
    """One activation per text. Returns `(features [n, hidden], kept_indices)`;
    a text that produced no activation (e.g. it tokenised to nothing) is skipped."""
    spec = {
        "extraction_points": [
            {
                "name": COLLECT_POINT,
                "layers": layer,
                "tensor_type": TENSOR_TYPE,
                "position": position,
                "probe_type": COLLECTOR_TYPE,
                "probe_kind": "single_shot",
            }
        ]
    }
    probes = {COLLECTOR_TYPE: ProbeFactory(ActivationCollectorProbe, {"max_tokens": 1})}
    rows, kept = [], []
    with ProbedModel.from_pretrained(model, spec=spec, probes=probes, device=device) as probed:
        for i, text in enumerate(texts):
            out = probed.generate(text, max_new_tokens=1, temperature=0.0)
            captured = out.probe_results[COLLECT_POINT].verdict
            if captured:
                rows.append(captured[0]["tensor"].float())
                kept.append(i)
            if (i + 1) % 100 == 0 or i + 1 == len(texts):
                print(f"  collected {i + 1}/{len(texts)}", flush=True)
    if not rows:
        raise RuntimeError(f"no activations captured at layer {layer}, position {position!r}; check the spec")
    return torch.stack(rows), kept


# --- splitting and training -----------------------------------------------------


def stratified_split(labels: Sequence[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    """Index lists for train/val/test, each class split 70/15/15 (val and test
    get at least one example of a class that has three or more)."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == cls]
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * SPLITS[1])) if len(idx) >= 3 else 0
        n_test = max(1, round(len(idx) * SPLITS[2])) if len(idx) >= 3 else 0
        val += idx[:n_val]
        test += idx[n_val : n_val + n_test]
        train += idx[n_val + n_test :]
    return train, val, test


def train_head(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    architecture: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> ProbeHead:
    """Logistic regression (or a one-hidden-layer MLP) with AdamW and a
    class-balanced BCE loss. Keeps the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    head = ProbeHead(x_train.shape[1], architecture, hidden_dim)
    with torch.no_grad():
        head.mean.copy_(x_train.mean(0))
        head.std.copy_(x_train.std(0, unbiased=False).clamp_min(1e-6))

    positives = float(y_train.sum())
    pos_weight = torch.tensor((len(y_train) - positives) / max(positives, 1.0))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    generator = torch.Generator().manual_seed(seed)

    best_loss, best_state = math.inf, None
    for epoch in range(epochs):
        head.train()
        for batch in torch.randperm(len(x_train), generator=generator).split(batch_size):
            optimizer.zero_grad()
            loss_fn(head(x_train[batch]), y_train[batch]).backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            x_eval, y_eval = (x_val, y_val) if len(x_val) else (x_train, y_train)
            val_loss = float(loss_fn(head(x_eval), y_eval))
        if best_state is None or val_loss < best_loss:
            best_loss, best_state = val_loss, {k: v.clone() for k, v in head.state_dict().items()}
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch + 1 == epochs:
            print(f"  epoch {epoch + 1}/{epochs}: val loss {val_loss:.4f}", flush=True)
    head.load_state_dict(best_state)
    head.eval()
    return head


# --- metrics -----------------------------------------------------------------------


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) statistic;
    tied scores get their average rank. NaN if only one class is present."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0  # 1-based average rank of each distinct score
    rank_sum = avg_rank[inverse][y].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores, dtype=np.float64) >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = auroc(labels, scores)
    return {
        "n": len(y),
        "accuracy": float((pred == y).mean()) if len(y) else math.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": None if math.isnan(auc) else auc,
    }


# --- main -------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    start = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[1/5] loading dataset {args.dataset!r}", flush=True)
    examples, label_names = load_examples(args.dataset, args.limit, args.seed)
    texts = [text for text, _ in examples]

    print(f"[2/5] collecting layer {args.layer} {TENSOR_TYPE} at {args.position} from {args.model} on {device}")
    features, kept = collect_activations(args.model, texts, layer=args.layer, position=args.position, device=device)
    labels = [examples[i][1] for i in kept]
    y = torch.tensor(labels, dtype=torch.float32)

    train_idx, val_idx, test_idx = stratified_split(labels, args.seed)
    print(f"[3/5] split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test", flush=True)

    print(f"[4/5] training a {args.probe} probe on {features.shape[1]} features", flush=True)
    head = train_head(
        features[train_idx],
        y[train_idx],
        features[val_idx],
        y[val_idx],
        architecture=args.probe,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    with torch.no_grad():
        scores = torch.sigmoid(head(features)).tolist()
    split_metrics = {
        name: binary_metrics([labels[i] for i in idx], [scores[i] for i in idx], args.threshold)
        for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))
    }
    for name, m in split_metrics.items():
        auc = "n/a" if m["auroc"] is None else f"{m['auroc']:.3f}"
        print(f"  {name:5s} n={m['n']:<5d} acc={m['accuracy']:.3f} f1={m['f1']:.3f} auroc={auc}")

    out = Path(args.out)
    print(f"[5/5] saving to {out}", flush=True)
    meta = {
        "layers": [args.layer],
        "tensor_type": TENSOR_TYPE,
        "position": args.position,
        "base_model": args.model,
        "label_names": list(label_names),
        "threshold": args.threshold,
        "dataset": args.dataset,
        "metrics": split_metrics,
    }
    save_probe(out, head, meta)
    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "layer": args.layer,
        "tensor_type": TENSOR_TYPE,
        "position": args.position,
        "probe": args.probe,
        "threshold": args.threshold,
        "seed": args.seed,
        "epochs": args.epochs,
        "num_examples": len(labels),
        "label_names": list(label_names),
        **split_metrics,
        "elapsed_s": round(time.monotonic() - start, 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"done in {metrics['elapsed_s']}s; run it with: python examples/train_probe/use_probe.py {out}")
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])

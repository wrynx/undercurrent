# Train a probe

This example goes from "I have a model" to "a trained probe runs inline in Undercurrent":

1. **Collect** one activation per labelled prompt with `ProbedModel` and
   `ActivationCollectorProbe` (`collector.py`).
2. **Train** a linear probe (logistic regression) or a small MLP with plain torch.
3. **Evaluate** it: accuracy, precision, recall, F1 and AUROC on held-out data.
4. **Save** it as `probe.safetensors` + `probe.json`.
5. **Load** it back as a `TrainedLinearProbe` that runs inline through `ProbedModel` and
   aborts generation when the score reaches a threshold (`use_probe.py`).

This is example code, not part of the `undercurrent` package. Clone the repository
and run it from the repository root. Probe save/load APIs, Hub upload and published
pretrained probes are on the [roadmap](../../ROADMAP.md), not in v0.1.

| File | What it is |
|---|---|
| `train_probe.py` | The training script: `python train_probe.py --help` |
| `use_probe.py` | About 20 lines: load a trained probe and generate with it |
| `linear_probe.py` | `TrainedLinearProbe`, `ProbeHead`, `save_probe`, `load_probe`, `read_meta`, `spec_for` |
| `collector.py` | `ActivationCollectorProbe`: stores detached CPU copies of activations |
| `data_sources.py` | `--dataset` name to `(text, label)` loader |
| `requirements.txt` | `datasets`, needed only for `--dataset civil_comments` |

## Quick run on CPU (toy dataset)

The `toy` dataset is 48 hand-written lines (24 benign, 24 insulting) built in code,
so nothing is downloaded except the model:

```bash
pip install undercurrent        # or, from a clone: pip install -e .
python examples/train_probe/train_probe.py --model gpt2 --dataset toy --layer 6 --epochs 20
python examples/train_probe/use_probe.py outputs/train_probe
```

On our development machine's CPU, the training script took 47 s with GPT-2, most of it
spent collecting activations. `use_probe.py` then prints one line per prompt, and the
insulting one should show `ABORTED`.

The toy set is tiny and templated, so its near-perfect scores only show that the
pipeline works. They say nothing about how well a probe detects toxicity.

## Real run on a GPU (Civil Comments)

```bash
pip install -r examples/train_probe/requirements.txt
python examples/train_probe/train_probe.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --dataset civil_comments --limit 4000 \
    --layer 16 --probe linear --epochs 20 --device cuda --out outputs/train_probe
python examples/train_probe/use_probe.py outputs/train_probe
```

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, not gated, 28 layers, hidden
  size 1536). Its weights take about 6 GB in float32, so it fits on a single GPU. Any decoder-only Hugging Face
  model with `model.layers` or `transformer.h` blocks works; `undercurrent
  inspect-model MODEL` lists the layers.
- **Layer:** a middle layer, about 55-65% of the way through the network. Probing
  work generally finds high-level features most linearly readable in the middle
  layers: early layers are still close to the tokens and the last few are
  specialised for predicting the next token. Try a few layers and compare val AUROC.
- **Runtime:** collection is one forward pass per example (`max_new_tokens=1`), so it
  grows linearly with `--limit`. Training a linear probe on a few thousand vectors
  takes seconds. We haven't timed a GPU run yet. The first run also streams the
  Civil Comments train split from the Hub.

## Dataset: Civil Comments

| | |
|---|---|
| Name | `google/civil_comments` |
| URL | <https://huggingface.co/datasets/google/civil_comments> |
| License | CC0 1.0 (public-domain dedication), per the dataset card |
| Size | 1.8M train / 97k validation / 97k test comments, each with crowd-rated `toxicity` in [0, 1] |

Preprocessing (`data_sources.py`):

- Streams the **train** split with `datasets.load_dataset(..., streaming=True)` and
  shuffles it through a 10,000-row buffer with `--seed`. Nothing is written to the
  repository; `datasets` caches in `~/.cache/huggingface`.
- Labels are binarised: `toxic` (1) if `toxicity >= 0.5`, else `non_toxic` (0).
- The sample is **class-balanced**: `--limit // 2` of each class (default `--limit`
  2000). Toxic comments are a small minority of the raw data.
- Empty comments are dropped, and comments are cut to their first 1000 characters
  to keep sequences short.

Civil Comments contains offensive text. Its labels come from crowd raters and carry
their biases; see the
[dataset card](https://huggingface.co/datasets/google/civil_comments) and the paper
(Borkan et al., 2019, arXiv:1903.04561).

## `train_probe.py` options

| Option | Default | Meaning |
|---|---|---|
| `--model` | `gpt2` | Hugging Face model id or local path |
| `--dataset` | `toy` | `toy` or `civil_comments` |
| `--layer` | (required) | Layer index to read (0-based) |
| `--position` | `prompt[-1]` | Spec position; the default is the last prompt token |
| `--limit` | all (toy) / 2000 | Number of examples, class-balanced |
| `--probe` | `linear` | `linear` (logistic regression) or `mlp` (one hidden layer, `--hidden-dim`) |
| `--epochs` | 20 | Training epochs (AdamW, `--lr 1e-3`, `--batch-size 64`) |
| `--threshold` | 0.5 | Score at which the probe flags; saved in `probe.json` |
| `--out` | `outputs/train_probe` | Output directory (gitignored) |
| `--device` | `cuda` if available, else `cpu` | Device for the model |
| `--seed` | 0 | Seeds the data sample, the split and training |

The data is split per class into 70% train, 15% validation and 15% test. Features are
standardised with the training-set mean and standard deviation, which are saved with
the weights. The loss is BCE weighted for class balance, and the script keeps the
epoch with the lowest validation loss. AUROC is computed with numpy (rank-sum
statistic); there's no scikit-learn dependency.

## Output format

`--out` gets three files:

- **`probe.safetensors`**: the `ProbeHead` state dict in float32, written with the
  `safetensors` library. Keys: `mean` and `std` (standardisation), plus `net.weight`
  and `net.bias` (linear) or `net.0.*` and `net.2.*` (MLP).
- **`probe.json`**: what you need to run the probe again:

  ```json
  {
    "format": "undercurrent-example-probe",
    "format_version": 1,
    "architecture": "linear",
    "input_dim": 1536,
    "layers": [16],
    "tensor_type": "residual_stream",
    "position": "prompt[-1]",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "label_names": ["non_toxic", "toxic"],
    "threshold": 0.5,
    "dataset": "civil_comments",
    "metrics": {"train": {...}, "val": {...}, "test": {...}}
  }
  ```

  MLP probes also record `hidden_dim`.
- **`metrics.json`**: the run's settings, plus `accuracy`, `precision`, `recall`,
  `f1`, `auroc` and `n` for each of `train`, `val` and `test`, and `elapsed_s`.

Loading checks `format_version`, the required keys and that `input_dim` matches the
weights, and fails with a message that says what to fix. Nothing is unpickled: the
weights are read with the `safetensors` loader, so loading a probe directory from
someone else can't run code.

## Using the trained probe

`use_probe.py` is the whole recipe:

```python
from linear_probe import PROBE_TYPE, load_probe, read_meta, spec_for
from undercurrent import ProbedModel

factory = load_probe("outputs/train_probe")  # ProbeFactory(TrainedLinearProbe, {head, threshold, ...})
meta = read_meta("outputs/train_probe")
with ProbedModel.from_pretrained(meta["base_model"], spec=spec_for(meta), probes={PROBE_TYPE: factory}) as m:
    out = m.generate("Shut up, you are a worthless idiot.", max_new_tokens=20)
print(out.aborted, out.probe_results["trained_probe"].verdict)
```

- `spec_for(meta)` builds an inline, `single_shot` extraction point at the layer,
  tensor type and position the probe was trained on. You can write the same thing in
  YAML, with `probe_type: trained_linear_probe`.
- The verdict is `{"score", "label", "flagged", "layer", "token_pos"}`, and each
  activation also emits a `ProbeSignal` with `confidence=score`. A score at or above
  the threshold returns `ABORT`, so `out.aborted` is True.
- To override the threshold for one extraction point, set `probe_args: {threshold:
  0.8}` in the spec.
- With the default `prompt[-1]` position the probe judges the prompt before
  generation. With the HF backend, an abort there still lets the first generated
  token through, because transformers checks its stopping criteria after each token.
  `train_probe.py` generates a single token per prompt, so that token never passes
  through the model: train on prompt positions (`prompt[-1]`, `prompt[0]`, ...).
- The probe has to run on the same base model, layer and tensor type it was trained
  on. A wrong hidden size fails with a clear error; a different model with the same
  hidden size won't fail, but its scores won't mean anything.

## Tests

`tests/examples/train_probe/` runs the whole pipeline on CPU and offline with a tiny
random GPT-2 (2 layers, `n_embd=32`) and the toy dataset, in under a minute:

```bash
pytest tests/examples/train_probe
```

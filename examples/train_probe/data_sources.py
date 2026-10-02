"""Labelled text for probe training: `--dataset NAME` -> list of `(text, label)`.

Named `data_sources` (not `datasets`) so it can't shadow the Hugging Face
`datasets` package on `sys.path`.

- `toy`: a few dozen hand-written lines, built in code. No download; for
  tests and smoke runs. A probe trained on it learns nothing useful.
- `civil_comments`: `google/civil_comments` from the Hugging Face Hub
  (CC0 1.0). Needs `pip install -r examples/train_probe/requirements.txt`
  and network access the first time.

Labels are binary: 1 is the positive ("flag this") class, `label_names[1]`.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

Example = tuple[str, int]


@dataclass(frozen=True)
class DataSource:
    name: str
    description: str
    label_names: tuple[str, str]
    load: Callable[[int | None, int], list[Example]]
    """`load(limit, seed)` -> shuffled examples; `limit=None` means the source's default size."""


# --- toy -----------------------------------------------------------------------

_TOY_SUBJECTS = ("the new library", "this recipe", "our team", "the weekend trip")
_TOY_BENIGN = (
    "I really enjoyed {s} and would recommend it.",
    "Thanks for sharing, {s} looks great.",
    "Could you tell me more about {s}?",
    "Honestly {s} was a pleasant surprise.",
    "I learned a lot from {s} today.",
    "We should talk about {s} at lunch.",
)
_TOY_TOXIC = (
    "Shut up, you are a worthless idiot and so is {s}.",
    "Only a pathetic moron would like {s}.",
    "You are stupid and {s} is garbage, get lost.",
    "Nobody cares, idiot, {s} is trash like you.",
    "What a disgusting loser you are, {s} is pathetic.",
    "Go away, you clueless fool, {s} is for idiots.",
)


def _load_toy(limit: int | None, seed: int) -> list[Example]:
    examples = [(t.format(s=s), 0) for t in _TOY_BENIGN for s in _TOY_SUBJECTS]
    examples += [(t.format(s=s), 1) for t in _TOY_TOXIC for s in _TOY_SUBJECTS]
    random.Random(seed).shuffle(examples)
    return _balanced(examples, limit) if limit is not None else examples


def _balanced(examples: list[Example], limit: int) -> list[Example]:
    """The first `limit` examples with the two classes as even as possible."""
    quota = [limit // 2, limit - limit // 2]
    out = []
    for text, label in examples:
        if quota[label] > 0:
            out.append((text, label))
            quota[label] -= 1
    return out


# --- civil_comments --------------------------------------------------------------

CIVIL_COMMENTS_ID = "google/civil_comments"
CIVIL_COMMENTS_DEFAULT_LIMIT = 2000
TOXICITY_THRESHOLD = 0.5
MAX_CHARS = 1000


def _load_civil_comments(limit: int | None, seed: int) -> list[Example]:
    """A class-balanced sample of the train split: `toxicity >= 0.5` is toxic.

    Streams the split (no full 400 MB download), shuffles it through a
    10k-row buffer and keeps `limit // 2` examples per class (toxic comments
    are a small minority of the split, so balancing matters). Comments are cut to
    `MAX_CHARS` characters to bound sequence length.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "--dataset civil_comments needs the Hugging Face `datasets` package: "
            "pip install -r examples/train_probe/requirements.txt"
        ) from exc

    limit = CIVIL_COMMENTS_DEFAULT_LIMIT if limit is None else limit
    quota = [limit // 2, limit - limit // 2]
    stream = load_dataset(CIVIL_COMMENTS_ID, split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    examples: list[Example] = []
    for row in stream:
        text = (row["text"] or "").strip()
        if not text:
            continue
        label = int(row["toxicity"] >= TOXICITY_THRESHOLD)
        if quota[label] > 0:
            examples.append((text[:MAX_CHARS], label))
            quota[label] -= 1
            if quota == [0, 0]:
                break
    random.Random(seed).shuffle(examples)
    return examples


SOURCES: dict[str, DataSource] = {
    "toy": DataSource("toy", "48 hand-written lines, built in code (offline)", ("non_toxic", "toxic"), _load_toy),
    "civil_comments": DataSource(
        "civil_comments",
        f"{CIVIL_COMMENTS_ID} (CC0 1.0), class-balanced, toxicity >= {TOXICITY_THRESHOLD}",
        ("non_toxic", "toxic"),
        _load_civil_comments,
    ),
}


def load_examples(name: str, limit: int | None = None, seed: int = 0) -> tuple[list[Example], tuple[str, str]]:
    """Load `(examples, label_names)` for the data source `name`."""
    source = SOURCES.get(name)
    if source is None:
        raise ValueError(f"unknown dataset {name!r}; choose one of: {', '.join(sorted(SOURCES))}")
    if limit is not None and limit < 2:
        raise ValueError(f"--limit must be at least 2 (one example per class), got {limit}")
    return source.load(limit, seed), source.label_names

"""PyTorch model stubs backing the content-safety probes.

Both modules are structural placeholders -- random-initialized weights
standing in for a trained safety classifier / trajectory scorer, exactly
like `undercurrent.core.examples`' deterministic stand-ins stand in for a real
model. The point of this package is validating that the platform wires
activations to a probe and a probe's signals back to the platform
correctly, not model quality; swap either module's internals for a trained
checkpoint without touching the `Probe` classes built on top of them.

Neither module uses `nn.Lazy*` layers: `input_dim` must be known at
construction time. This is deliberate -- a Lazy layer only materializes
(and randomly initializes) its weights on its *first forward call*, not at
construction, which would make seeding for reproducible tests fragile (the
seed would need to be held across an arbitrary gap between construction and
first use, spanning whatever other torch RNG consumption happens in
between). Both `SingleTokenSafetyProbe` and `TrajectorySafetyProbe` instead
defer *building* these modules until they see their first real activation
and know its flattened dimension, then construct with a seed scoped tightly
around that one construction call. See their `_build_*` methods.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class SafetyClassifierHead(nn.Module):
    """single_shot classifier head: one activation vector -> one safety score in [0, 1].

    `hidden_sizes` is either a single int (one hidden layer, `Linear ->
    activation -> Linear` -- the shape `single_token.py`'s random-init wiring
    stub uses) or a sequence of ints for a deeper stack (`Linear -> activation
    -> [Dropout] ->` per hidden size, then a final `Linear` to `output_dim`
    logits) -- the shape a real externally-trained checkpoint loaded by
    `custom_mlp.py` may use. `activation` selects the nonlinearity after each
    hidden `Linear` ("relu" or "gelu") -- this can't be recovered from a
    checkpoint's tensor shapes (activations carry no learned parameters), so
    it must be passed explicitly to match how the checkpoint was trained.
    `output_dim=1` (the default) is a single logit passed through `sigmoid`;
    `output_dim>1` is treated as class logits passed through `softmax`, with
    the last class's probability reported as the score, so a checkpoint
    trained with an N-way softmax head round-trips through the same [0, 1]
    "score" contract as the single-logit case. `final_activation=True` applies
    `activation` to the output logits before `sigmoid`/`softmax` -- some
    training code runs the same activation module across every layer of
    `net`, output layer included, before a separate final `Softmax`; this
    reproduces that exactly rather than approximating it. `dropout`
    of 0 (the default) omits the `Dropout` modules entirely, which is what
    keeps the single-hidden-layer case's `net` indices (`net.0`, `net.2`)
    unchanged.
    """

    _ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: int | Sequence[int] = 32,
        dropout: float = 0.0,
        output_dim: int = 1,
        activation: str = "relu",
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(self._ACTIVATIONS)}, got {activation!r}")
        self.output_dim = output_dim
        self.final_activation = final_activation
        activation_cls = self._ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size
        layers.append(nn.Linear(prev_dim, output_dim))
        if final_activation:
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x.unsqueeze(0)).squeeze(0)
        if self.output_dim == 1:
            return torch.sigmoid(logits).squeeze(-1)
        # two-or-more-way softmax head: report the probability mass on the
        # last class, which is the "unsafe"/positive class by training
        # convention, as the same [0, 1] score a 1-unit sigmoid head returns.
        return torch.softmax(logits, dim=-1)[..., -1]


class TrajectoryRecurrence(nn.Module):
    """trajectory scorer: a GRU-cell recurrence folding each new activation
    into a running hidden state, producing a safety score in [0, 1] per
    step. The GRU's own gating is what gives the running score its
    exponentially-weighted-ish memory of past steps -- there is no separate
    EWMA blend on top of it.
    """

    def __init__(self, input_dim: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.score_head = nn.Linear(hidden_size, 1)

    def initial_hidden(self) -> torch.Tensor:
        return torch.zeros(self.hidden_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_proj(x.unsqueeze(0))
        new_hidden = self.cell(projected, hidden.unsqueeze(0)).squeeze(0)
        score = torch.sigmoid(self.score_head(new_hidden)).squeeze(-1)
        return score, new_hidden

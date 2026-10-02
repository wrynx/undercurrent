"""Example Probe implementations validating the undercurrent.core interface.

These are reference/test fixtures, not production probes: both use
deterministic stub scoring in place of a real model. They demonstrate the
two `probe_kind` shapes the interface needs to support:

- `MLPClassifierProbe` (single_shot): one activation in, one verdict out.
- `TrajectoryScoreProbe` (trajectory): running state across many
  activations, with a mid-request abort path via `check_intervention`.
"""

from .mlp_classifier import MLPClassifierProbe
from .trajectory_score import TrajectoryScoreProbe

__all__ = ["MLPClassifierProbe", "TrajectoryScoreProbe"]

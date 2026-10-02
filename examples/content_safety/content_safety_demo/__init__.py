"""content_safety_demo: demo content-safety probes for Undercurrent.

Lives in examples/content_safety/ and is not installed with the package.

Built on `undercurrent.core` (`Probe`, `ProbeFactory`/`.spawn()`, `ProbeSignal`,
`ProbeResult`, `RequestContext`) and `undercurrent.spec` (`ActivationRecord`,
`ExtractionPoint`) -- imported directly from those packages, never
redefined. Exists as the platform's reference test case: it demonstrates
both `probe_kind` variants end-to-end using structural (random-initialized)
PyTorch models, not trained ones -- correctness of the platform wiring is
what this package validates, not model quality.

- `SingleTokenSafetyProbe` (probe_kind="single_shot"): one activation in,
  one safety verdict out, via a small linear/MLP classifier head.
- `TrajectorySafetyProbe` (probe_kind="trajectory"): a running safety score
  across every matching activation in a request, via a small GRU-cell
  recurrence, aborting once the score crosses a threshold.
- `CustomMLPProbe` (probe_kind="single_shot"): like `SingleTokenSafetyProbe`,
  but loads a REAL trained classifier-head checkpoint from disk (path given
  at construction time) instead of using random-initialized weights -- see
  `custom_mlp.py`.

`spec_binding` binds `undercurrent.spec.ExtractionPoint` config (e.g. parsed from
a YAML spec) to these classes -- see `examples/content_safety_llama.yaml`
for a sample spec, `tests/test_live_llama_integration.py` for how the
resulting wiring is registered against a real adapter/model, and
`examples/serve_llama_mlp_pipeline.py` for `CustomMLPProbe` attached to a
live Llama-3.1-8B model behind an HTTP server.
"""

from .custom_mlp import CustomMLPProbe
from .models import SafetyClassifierHead, TrajectoryRecurrence
from .single_token import SingleTokenSafetyProbe
from .spec_binding import probe_kwargs_from_extraction_point, resolve_probe_cls
from .trajectory import TrajectorySafetyProbe

__all__ = [
    "CustomMLPProbe",
    "SafetyClassifierHead",
    "SingleTokenSafetyProbe",
    "TrajectoryRecurrence",
    "TrajectorySafetyProbe",
    "probe_kwargs_from_extraction_point",
    "resolve_probe_cls",
]

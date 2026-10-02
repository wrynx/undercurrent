"""`vllm.general_plugins` entry point: `register()`.

Why this exists
----------------
`VLLMEngineAdapter.load_model()` (see `adapter.py`) is the fully-wired path
for using this adapter *embedded in a Python process you control* -- it
builds `AsyncEngineArgs` itself, so it can pass
`worker_extension_cls="undercurrent.adapters.vllm.worker_extension.ProbingWorkerExtension"`
directly and force `VLLM_USE_V2_MODEL_RUNNER=0` /
`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` before constructing the engine.

That path doesn't exist for a bare `vllm serve <model> ...` CLI invocation --
there is no `VLLMEngineAdapter.load_model()` call for a plugin to hook into,
because `vllm serve` builds `EngineArgs`/`VllmConfig` itself from CLI flags.
This module is `undercurrent.adapters.vllm`'s hook into *that* startup path, via
vLLM's `vllm.general_plugins` entry-point mechanism
(see `vllm/plugins/__init__.py`'s `load_general_plugins()`).

What we confirmed against the actual installed vLLM 0.28.0 source (not
docs) before writing this, and what that implies for what `register()` can
safely do:

- `load_general_plugins()` calls every registered `vllm.general_plugins`
  entry point with **zero arguments**, and it runs in **every** process
  vLLM starts for a single `vllm serve` invocation: the API-server/frontend
  process, the `EngineCore` process (confirmed call site:
  `vllm/v1/engine/core.py`), and every TP/PP worker subprocess (confirmed
  call site: `vllm/v1/worker/worker_base.py`). `register()` should therefore
  be treated as "may run several times, in several unrelated processes,
  with no way to tell them apart except best-effort heuristics" -- see
  `_process_role()` below.
- Because it's called with zero arguments, `register()` is never handed a
  `VllmConfig`/`EngineArgs` object to mutate, and by the time any plugin
  runs, `EngineArgs.worker_extension_cls`'s default has already been baked
  in for a bare `vllm serve` invocation. There is no supported way to set
  `worker_extension_cls` from here -- it must stay a user-supplied
  `--worker-extension-cls undercurrent.adapters.vllm.worker_extension.ProbingWorkerExtension`
  CLI flag (or the equivalent key in a `--config some.yaml`). Anything that
  tried to reach into vLLM's config objects from a zero-arg plugin hook
  would be relying on undocumented internals for something vLLM's own
  plugin API deliberately doesn't expose -- not attempted here.
- What a zero-arg, multiply-invoked hook *can* safely do is idempotent,
  process-global, side-effect-only setup that doesn't need any engine
  context: logging, and -- see `_apply_env_defaults()` -- setting process
  environment variables that vLLM itself later reads via `os.environ` at
  config/engine-construction time, which for every confirmed call site
  above happens strictly *after* plugin loading. That ordering (plugins
  load before engine/config construction reads these particular variables,
  in each process that has its own copy of `os.environ`) is what makes it
  safe to set them here rather than redundant or too-late -- see that
  function's docstring for the two specific variables and why each is
  chosen, and why we only ever set-if-unset instead of forcing a value the
  way `adapter.py`'s `load_model()` does (which has an explicit opt-out
  kwarg available to callers; a zero-arg plugin hook doesn't).

Internal and experimental: this reads undocumented vLLM internals and is
not part of the public API (see docs/api-stability.md); it changes whenever
vLLM does.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Guards against `register()` doing its logging/env-setdefault work more than
# once per process. vLLM's plugin loader can invoke a given entry point
# multiple times within the *same* process too (e.g. re-imports during
# argument parsing vs. actual engine startup) -- not just once per process as
# the module docstring's "every process" framing might suggest -- so this is
# a per-process, not per-call, guard.
_REGISTERED = False


def _process_role() -> str:
    """Best-effort guess at which of vLLM's processes we're running in, for
    log messages only -- never used for control flow. There is no supported
    API to ask vLLM "what process am I" from a zero-arg plugin hook, so this
    is deliberately heuristic and allowed to say "unknown".
    """
    argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
    if "VLLM_WORKER_MULTIPROC_METHOD" in os.environ or "engine_core" in " ".join(sys.argv):
        return "engine-core-or-worker"
    if argv0 in ("vllm",) or "serve" in sys.argv[:2]:
        return "api-server/frontend"
    return "unknown"


def _apply_env_defaults() -> None:
    """Set-if-unset the two env vars `adapter.py`'s `load_model()` already
    forces for the embedded-adapter usage path, so a bare `vllm serve`
    invocation gets the same correctness defaults instead of silently
    mis-capturing (see `adapter.py`'s inline comments for the full
    per-variable rationale; summarized here for why it's *also* correct to
    set them from a zero-arg plugin hook):

    - `VLLM_USE_V2_MODEL_RUNNER=0` -- this adapter's `introspection.py` only
      understands vLLM's V1 `GPUModelRunner` internals; some vLLM
      versions/environments default to (or already have an env-set) V2
      runner even for architectures that don't need it, which breaks every
      forward hook. Reading this env var happens when `VllmConfig` /
      the model runner is constructed -- strictly after plugin loading at
      every confirmed call site (see module docstring) -- so setting it here
      is not too late.
    - `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` -- some vLLM versions
      rewrite `request_id` before it reaches the scheduler, which breaks
      this adapter's row -> (request_id, token_pos) mapping. Read at
      request-processing time, also after plugin loading.

    Unlike `adapter.py`'s `load_model()` (which has explicit
    `allow_v2_model_runner=`/`allow_request_id_randomization=` kwargs a
    caller can pass to opt out), this hook has no caller-supplied
    configuration at all -- so it only fills in a value if the operator
    hasn't already set one themselves (`os.environ.setdefault`, not
    assignment), rather than overriding an explicit choice made via the
    shell environment `vllm serve` was launched in.
    """
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")


def register() -> None:
    """Entry point for `vllm.general_plugins`. Called with zero arguments by
    vLLM's plugin loader, potentially multiple times, in every process a
    `vllm serve` invocation starts (API-server/frontend, `EngineCore`, and
    each TP/PP worker subprocess).

    Deliberately modest: this does NOT set `worker_extension_cls` and does
    NOT touch any vLLM config object -- both are confirmed unsupported from
    a zero-arg plugin hook (see module docstring). You still must pass
    `--worker-extension-cls undercurrent.adapters.vllm.worker_extension.ProbingWorkerExtension`
    (or the config-file equivalent) explicitly to `vllm serve` yourself;
    installing this package and its entry point does not do that for you.
    """
    global _REGISTERED
    role = _process_role()
    if _REGISTERED:
        logger.debug(
            "undercurrent.adapters.vllm plugin register() called again in this process (role=%s); no-op.", role
        )
        return
    _REGISTERED = True

    logger.info(
        "undercurrent.adapters.vllm vllm.general_plugins entry point loaded (process role guess: %s). "
        "Remember: --worker-extension-cls "
        "undercurrent.adapters.vllm.worker_extension.ProbingWorkerExtension is still required "
        "for activation capture to actually be wired up -- this plugin does not set it for you.",
        role,
    )
    _apply_env_defaults()

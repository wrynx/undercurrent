# Contributing to Undercurrent

Thanks for helping build Undercurrent, Wrynx's activation-probing platform. This guide covers
everything from cloning the repo to getting a pull request merged. For how the library works, see
the [documentation site](https://wrynx.github.io/undercurrent/).

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). How decisions
are made and how to become a maintainer is described in [GOVERNANCE.md](GOVERNANCE.md).

## Ways to contribute

- **Questions and ideas:** use [GitHub Discussions](https://github.com/wrynx/undercurrent/discussions).
  Issues are for bugs and concrete proposals.
- **Bug reports and feature requests:** open an [issue](https://github.com/wrynx/undercurrent/issues/new/choose)
  using one of the templates.
- **Docs:** fixes and new examples are very welcome. See [docs/README.md](docs/README.md) for how
  the site is built and which code blocks CI executes.
- **Probes, sinks and engine adapters:** see the checklists [below](#adding-a-probe). For a new
  adapter, or anything that changes public API, please open an issue first so we can agree on the
  design before you write a lot of code.
- **Security issues:** do **not** open a public issue or PR. Follow [SECURITY.md](SECURITY.md).

## Development setup

You need Python 3.10 or newer and git. One block takes you from a fresh clone to a working
environment:

```bash
git clone https://github.com/wrynx/undercurrent.git && cd undercurrent
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu && pre-commit install
```

- The base install already includes the Hugging Face backend (`torch`, `transformers`), so the
  whole CPU test suite runs with nothing else installed. The `--extra-index-url` points pip at
  PyTorch's CPU-only wheels, which are much smaller than the default CUDA ones on Linux. On macOS
  it's harmless.
- **vLLM work** (`src/undercurrent/adapters/vllm/`) needs a machine with a CUDA GPU and a vLLM
  install that matches its CUDA and torch versions. Either run `pip install -e ".[dev]"` inside your
  existing vLLM environment, or install the tested range with `pip install -e ".[dev,vllm]"`
  (GPU only; not verified as part of this guide). Supported versions are in
  [docs/compatibility.md](docs/compatibility.md).
- The `dev` extra is for contributors only and installs pytest, ruff, mypy and pre-commit.

## Running tests

```bash
pytest                                   # the full suite
pytest tests/core/test_registry.py       # one file
pytest tests/core/test_registry.py::test_decorator_registers_and_returns_class  # one test
pytest -m "not gpu and not network"      # exactly what CI runs
pytest -m "not slow"                     # skip tests that take more than ~10s
RUN_NETWORK_TESTS=1 pytest -m network    # tests that download from the Hugging Face Hub
```

Coverage is opt-in (it is not in the default `addopts`). To get the same report CI uploads to Codecov:

```bash
pytest -m "not gpu and not network" --cov --cov-report=term --cov-report=xml   # writes coverage.xml
```

The coverage settings are in `[tool.coverage.*]` in `pyproject.toml`. Code that only runs against a live vLLM
engine is marked `# pragma: no cover - needs a live vLLM engine`; use that marker only for code no CPU test can
reach.

Tests are marked (see `[tool.pytest.ini_options]` in `pyproject.toml` and `tests/conftest.py`):

| Marker    | Meaning                                    | Default behaviour                              |
|-----------|--------------------------------------------|------------------------------------------------|
| `gpu`     | needs a CUDA GPU                           | skipped automatically when CUDA is unavailable |
| `network` | downloads from the Hub or the internet     | skipped unless `RUN_NETWORK_TESTS=1`           |
| `slow`    | takes more than ~10s                       | runs                                           |

On a laptop, plain `pytest` is therefore the same as the CI command. Tests must not reach the
network unless they're marked `network`; CI sets `HF_HUB_OFFLINE=1` to catch that.

**CI** (`.github/workflows/ci.yml`) runs ruff, mypy, and the CPU suite on Python 3.10-3.13 (Linux)
and 3.12 (macOS) for every pull request. There is no GPU CI yet: the GPU and real-vLLM tests run
on a maintainer's GPU machine with `scripts/gpu_check.sh` (which you can also run yourself on any
CUDA machine with vLLM installed). If your change touches vLLM or CUDA code, say so in the PR
description, and say whether you ran `scripts/gpu_check.sh`.

## Lint, format and type-check

`pre-commit install` (in the setup block) runs ruff and a few file checks on every commit. To run
the same checks by hand:

```bash
pre-commit run --all-files   # everything the hooks check
ruff check --fix .           # lint, with safe autofixes
ruff format .                # format
mypy                         # type-check src/undercurrent (settings in pyproject.toml)
```

CI runs `ruff check .`, `ruff format --check .` and `mypy`. mypy is stricter on `spec`, `core`,
`router` and `sinks` (full annotations required) than on the adapters, which hook untyped ML
libraries. The target is Python 3.10, so don't use newer syntax or stdlib APIs without a fallback.

If `mypy` stops with `numpy/__init__.pyi: error: Type statement is only supported in Python 3.12
and greater`, your environment has numpy 2.5 or newer, whose stubs mypy can't read when targeting
3.10. Run `pip install "numpy<2.5"` and re-run `mypy`.

## Previewing the docs

```bash
pip install -e ".[docs]" --extra-index-url https://download.pytorch.org/whl/cpu
mkdocs serve             # live preview at http://127.0.0.1:8000/
mkdocs build --strict    # what CI runs; any warning fails the build
```

The navigation in `mkdocs.yml` is fixed; read [docs/README.md](docs/README.md) before adding pages
or code blocks.

## Signing off your commits (DCO)

Undercurrent uses the [Developer Certificate of Origin](https://developercertificate.org/) (DCO)
instead of a CLA. By signing off a commit you certify that you wrote the change, or otherwise have
the right to submit it under the project's license (Apache-2.0). You keep your copyright.

Sign off **every** commit with `-s`:

```bash
git commit -s -m "Fix off-by-one in generated[*] position matching"
```

This adds a trailer with the name and email from your git config, which must be real:

```text
Signed-off-by: Jane Doe <jane@example.com>
```

A DCO check runs on every pull request and blocks merging until every commit is signed off. If you
forgot:

```bash
git commit --amend --no-edit -s       # only the last commit
git rebase --signoff main             # every commit on your branch since main
git push --force-with-lease           # update the PR
```

(Use `upstream/main` or `origin/main` instead of `main` if your local `main` is out of date.)

## Pull request checklist

- [ ] **Focused scope.** One logical change per PR. Split unrelated fixes and refactors out.
- [ ] **Tests** for the behaviour you add or change, passing locally with `pytest`.
- [ ] **`pre-commit run --all-files`** and **`mypy`** pass.
- [ ] **Docs** updated: docstrings (Google style), the relevant page under `docs/`, and the README
      if the quickstart changes.
- [ ] **Changelog:** for user-facing changes, add an entry under `## [Unreleased]` in
      [CHANGELOG.md](CHANGELOG.md) (Added / Changed / Fixed / Removed; breaking changes say what
      users must do).
- [ ] **Signed off:** every commit has a `Signed-off-by` line.
- [ ] **GPU:** if you touched vLLM or CUDA paths, say so, and whether you ran `scripts/gpu_check.sh`.

The [pull request template](.github/PULL_REQUEST_TEMPLATE.md) repeats this list. A PR needs approval
from at least one maintainer who didn't write it ([GOVERNANCE.md](GOVERNANCE.md)).

## Adding a probe

A probe subclasses `undercurrent.core.Probe`, sets `probe_kind` (`"single_shot"` or `"trajectory"`)
and implements `on_start`, `on_activation` and `on_end`. Instances are spawned fresh per request, so
keep per-request state on `self`, never in class attributes (they're rejected at class definition).
The [custom probe guide](https://wrynx.github.io/undercurrent/guides/custom-probe/) covers the API.

- **A probe shipped with the library:** add it under `src/undercurrent/core/examples/`, next to
  `mlp_classifier.py` and `trajectory_score.py`. To let specs refer to it by `probe_type`, register
  it with `@register_probe("my_probe")` (from `undercurrent.core`). Copy the tests from
  `tests/core/test_trajectory_score_probe.py`; use the `make_record` / `make_request_ctx` fixtures
  in `tests/core/conftest.py`.
- **A demo or domain-specific probe:** put it under `examples/<name>/` with a README and a YAML spec.
  Examples aren't installed; test them in `tests/examples/<name>/` with a `conftest.py` that adds
  the example directory to `sys.path` (copy `tests/examples/openai_server/conftest.py`).
- **A probe in your own package:** you don't need a PR at all. Publish it and declare an entry point
  in the `undercurrent.probes` group (`my_probe = "my_package.probes:MyProbe"`); specs can then
  refer to it by `probe_type`. See the module docstring of `src/undercurrent/core/registry.py`.

## Adding a sink

Observation sinks receive async probes' signals and results.

1. Subclass `undercurrent.sinks.LogSink` in `src/undercurrent/sinks/<name>_sink.py` and implement
   `write_signal(request_id, extraction_point_name, signal)` and
   `write_result(request_id, extraction_point_name, result)`.
2. Neither method may block for long: they're called from router worker threads. Hand network I/O
   to a background worker, as `webhook_sink.py` does.
3. Accept `redact=` and call `super().__init__(redact=redact)`; pass every record through
   `self._apply_redaction(...)` before writing it (see `file_sink.py`).
4. Export it from `src/undercurrent/sinks/__init__.py`.
5. Tests go in `tests/sinks/test_<name>_sink.py`; copy `test_file_sink.py` (local writes) or
   `test_webhook_sink.py` (network, retries, dead-lettering), and `test_router_integration.py` for
   wiring through a `Router`.
6. Document it in `docs/guides/observation-sinks.md`; `docs/reference/sinks.md` picks up the
   docstrings.

## Adding an engine adapter

Open an issue with the "New engine adapter" template first. Adapters hook engine internals and are
the most expensive code to maintain.

1. Create `src/undercurrent/adapters/<engine>/` with a class that subclasses
   `undercurrent.adapters.base.EngineAdapter` and implements `load_model`, `register_extraction`,
   `generate` and `unregister_extraction`. The contract (isolation between requests, honouring an
   inline `abort`, cleanup after abort) is in the docstring of `src/undercurrent/adapters/base.py`.
2. Start from `adapters/hf/` (the reference implementation: a sequential decode loop and forward
   hooks). Look at `adapters/vllm/` for a continuously-batched engine (worker-side capture, sequence
   mapping, a runtime version check).
3. Never import the engine at module import time. Use the lazy `require_*()` helpers in
   `src/undercurrent/adapters/_optional.py` (add one for your engine) so `import undercurrent` stays
   light. Don't add the engine to the base dependencies.
4. Tests go in `tests/adapters/<engine>/`. Copy `tests/adapters/hf/test_adapter_contract.py` for the
   contract tests and `tests/adapters/hf/_helpers.py` for tiny random-weight models that need no
   download. Anything that needs a GPU or real weights is marked `gpu` and/or `network` and calls
   `pytest.importorskip("<engine>")`, like `tests/adapters/vllm/test_integration_vllm.py`.
5. Document the adapter's own limits (supported tensor types, intervention latency) in its module
   docstring and in `docs/reference/adapters.md`, and add the supported engine versions to
   `docs/compatibility.md`.

## Compatibility, releases and security

- **Compatibility policy:** supported dependency ranges and how they're tested are in
  [docs/compatibility.md](docs/compatibility.md). Changing a range needs a matching update there.
- **Releases:** maintainers cut releases from git tags; see [RELEASING.md](RELEASING.md).
- **Security:** report vulnerabilities privately to [security@wrynx.com](mailto:security@wrynx.com)
  as described in [SECURITY.md](SECURITY.md).

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).

# Releasing Undercurrent

This is the maintainer checklist for cutting a release of `undercurrent`. A
release is a signed git tag. Pushing the tag starts
`.github/workflows/release.yml`, which builds, tests and publishes the
package. Nobody uploads to PyPI by hand, and the workflow needs no API tokens.

## How versions and tags work

- The version comes from the git tag via `hatch-vcs`. The tag `v0.1.0`
  builds version `0.1.0`, and the tag `v0.1.0rc1` builds `0.1.0rc1`. There
  is no version string to edit anywhere in the source tree.
- Tags are `v` followed by a [PEP 440](https://peps.python.org/pep-0440/)
  version: `vX.Y.Z` for a final release, and `vX.Y.ZrcN`, `vX.Y.ZaN` or
  `vX.Y.ZbN` for pre-releases. Don't use other forms such as `v0.1.0-rc.1`.
- Untagged commits build as dev versions such as `0.1.devN+g<sha>`, which
  are never published.
- Building needs the git history and tags. A shallow clone, or a tree with no
  `.git`, builds as version `0.0`. To build a specific version without git,
  set `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0` (local testing only).

## What the release workflow does

`release.yml` runs on every pushed tag that matches `v*`:

1. **build**: checks out the full history, runs `python -m build`, fails if
   the built version isn't the tag without its `v`, and runs `twine check`.
2. **smoke**: installs the built wheel and sdist into fresh virtualenvs on
   Python 3.10 and 3.13 with CPU-only torch, and runs
   `scripts/smoke_test_wheel.sh` against them.
3. **publish-testpypi** or **publish-pypi**: uses PyPI Trusted Publishing
   (OIDC). Exactly one of the two runs, decided by
   `packaging.version.Version(tag).is_prerelease` in the build job:
   - pre-release tags (`v0.1.0rc1`, `v0.2.0b2`) go to **TestPyPI** in the
     `publish-testpypi` job, through the GitHub environment `testpypi`
   - final tags (`v0.1.0`) go to **PyPI** in the `publish-pypi` job, through
     the GitHub environment `pypi`, which needs a maintainer's approval
4. **github-release**: creates the GitHub Release with the built files
   attached, marked as a pre-release for pre-release tags. If `CHANGELOG.md`
   has a section for the version, that section becomes the release notes.
   Otherwise GitHub generates notes. Pre-release tags (`0.1.0rc1`) have no
   changelog section of their own, so they get generated notes.

A pre-release tag can never publish to PyPI, and a final tag never goes only
to TestPyPI. So the release process is: tag a release candidate, check it on
TestPyPI, then tag the final release from the same commit.

## One-time setup

These are done once per repository and are already in place for normal
releases. Check them before the first release.

- On both [PyPI](https://pypi.org/manage/account/publishing/) and
  [TestPyPI](https://test.pypi.org/manage/account/publishing/), the project
  `undercurrent` has a trusted publisher with owner `wrynx`, repository
  `undercurrent`, workflow `release.yml`, and environment `pypi` (PyPI) or
  `testpypi` (TestPyPI).
- The GitHub repository has the environments `testpypi` and `pypi`. The
  `pypi` environment has a required reviewer.
- You can sign tags (`git config user.signingkey`, GPG or SSH), and your
  public key is on your GitHub account so the tag shows as verified.

## 1. Pre-flight checklist

Do all of this before tagging anything:

- [ ] CI is green on the `main` commit you're going to tag.
- [ ] `scripts/gpu_check.sh` passes on a GPU machine, on the commit you're
      going to tag. There is no GPU CI for v0.1, so this is the only run of
      the GPU and real-vLLM tests. See [The GPU check](#the-gpu-check). If it
      fails, either fix it or record the known issue in the changelog.
- [ ] `CHANGELOG.md` is up to date:
  - Move everything under `## [Unreleased]` into a new
    `## [X.Y.Z] - YYYY-MM-DD` section, dated with the day you plan to tag
    the final release.
  - Leave an empty `## [Unreleased]` heading above it.
  - Update the compare links at the bottom: `[Unreleased]` compares
    `vX.Y.Z...HEAD`, and `[X.Y.Z]` compares the previous tag with `vX.Y.Z`.
  - Breaking changes are listed under **Changed** or **Removed**, and each
    says what users need to do.
- [ ] `CITATION.cff` is up to date: set `version` to `X.Y.Z` (the final
      version, without the `v`, also when you tag a release candidate) and
      `date-released` to the same date as the `CHANGELOG.md` section. They
      aren't derived from the tag.
- [ ] `docs/compatibility.md` matches `pyproject.toml` (dependency ranges,
      the supported vLLM range, `SUPPORTED_VLLM`) and lists only
      combinations that have actually been tested.
- [ ] Those changes are merged to `main`, and your local `main` is that
      commit:

```bash
git switch main
git pull --ff-only origin main
git status            # clean
git log -1 --oneline  # the commit you're about to tag
```

### The GPU check

`scripts/gpu_check.sh` runs the `gpu`-marked tests and the vLLM adapter
tests (`tests/adapters/vllm`) against real vLLM, the same tests as the
`GPU` workflow. You need any Linux machine with a CUDA GPU and a vLLM in the
supported range (`SUPPORTED_VLLM`, see `docs/compatibility.md`). Install
Undercurrent the documented production way, into the existing vLLM
environment, from a checkout of the commit you're going to tag:

```bash
git checkout <commit>                 # the commit you're about to tag
pip install "vllm>=0.28,<0.29"        # skip if the environment already has it
pip install -e ".[dev]"               # not [vllm]: vLLM is already there
scripts/gpu_check.sh
```

The script installs nothing. It downloads `gpt2` from the Hugging Face Hub,
and it fails if CUDA or vLLM isn't usable, if the vLLM is outside the
supported range, if a test fails, or if any GPU test was skipped. Add
`--live-llama` (with `UNDERCURRENT_LIVE_LLAMA_MODEL` set) to also run the live
Llama content-safety test.

The GPU tests include `tests/adapters/vllm/test_residual_stream_gpu.py`,
which loads a tiny random Llama and a tiny random GPT-2 with both backends
and checks that vLLM's `residual_stream` captures match the HF backend's,
token by token (the script fails if it didn't run). It is the GPU
verification of the 0.1.0 fix for `residual_stream` on fused-residual
layers. The first time it passes, change "GPU-verified: pending" in
`docs/compatibility.md` (Known issues) and the matching note in
`CHANGELOG.md` to the date and the combination it ran on.

At the end it prints the Python, torch, transformers, vLLM, CUDA and GPU it
ran with. Add that combination to the tested-combinations table in
`docs/compatibility.md` and merge it to `main` before tagging. A commit that
only changes docs or `CHANGELOG.md` doesn't need another GPU check; any code
or packaging change does.

If a maintainer has registered a self-hosted GPU runner, running the `GPU`
workflow by hand (Actions → GPU → Run workflow) on that commit does the same
check.

## 2. Tag a release candidate

Use the version you're releasing (`0.1.0` here) with `rc1`. If you need
another candidate later, use `rc2`, `rc3` and so on. Never reuse a tag.

```bash
git tag -s v0.1.0rc1 -m "v0.1.0rc1"
git push origin v0.1.0rc1
```

Watch the run at https://github.com/wrynx/undercurrent/actions. Approve the
`testpypi` environment if it asks. When the run finishes, the release
candidate is at https://test.pypi.org/project/undercurrent/.

If the **build** job fails the version check, the tag isn't on the commit
you expected, or the checkout had no tags. Nothing was published. Leave the
tag where it is, fix the problem, and tag the next candidate (`rc2`). See
[If something goes wrong](#if-something-goes-wrong).

## 3. Verify the release candidate on TestPyPI

Use a fresh virtualenv on Python 3.10 (the oldest supported version) outside
the repository checkout, so you test the published package and not your
source tree. Install CPU-only torch first to avoid downloading multi-GB CUDA
wheels:

```bash
python3.10 -m venv /tmp/uc-rc && source /tmp/uc-rc/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ undercurrent==0.1.0rc1
python -c "import undercurrent; print(undercurrent.__version__)"   # 0.1.0rc1
```

`--extra-index-url` lets pip get the dependencies (pydantic, transformers,
...) from the real PyPI, because TestPyPI doesn't host them.

Then run the smoke test. It loads GPT-2 (downloaded from the Hugging Face
Hub), captures an activation with the Hugging Face adapter and generates a
few tokens on CPU:

```bash
cd /tmp
cat > uc_smoke.py <<'EOF'
import undercurrent
from undercurrent.adapters.hf import HFEngineAdapter
from undercurrent.core import ProbeFactory
from undercurrent.core.examples import MLPClassifierProbe
from undercurrent.router import Router
from undercurrent.spec import parse_yaml

print("undercurrent", undercurrent.__version__)
spec = parse_yaml("""
version: "1"
extraction_points:
  - name: smoke
    layers: 6
    tensor_type: residual_stream
    position: "prompt[-1]"
    probe_type: linear_probe
    probe_kind: single_shot
    execution_mode: inline
""")
router = Router(probe_registry={"linear_probe": ProbeFactory(MLPClassifierProbe, {"num_classes": 2})})
adapter = HFEngineAdapter()
adapter.load_model("gpt2")
adapter.register_extraction("smoke", list(spec))
print(adapter.generate("smoke", "Hello, my name is", {"max_new_tokens": 8}, router))
router.shutdown()
EOF
python uc_smoke.py
```

It should print the version and a short continuation of the prompt, with no
traceback. Also:

- [ ] Run the quickstart from the README of the tagged commit.
- [ ] Check https://test.pypi.org/project/undercurrent/: the README renders,
      the project links work, and the classifiers and Python versions are
      right.
- [ ] Check the GitHub Release for `v0.1.0rc1`: it's marked as a
      pre-release and has the wheel and sdist attached.
- [ ] Optional, if you have a GPU machine with vLLM in the supported range:
      `pip install` the release candidate into that environment and check
      that constructing `VLLMEngineAdapter` passes the version check.

If anything is wrong, fix it on `main` and go back to step 2 with the next
candidate (`rc2`).

## 4. Tag the final release

Tag the **same commit** as the release candidate you verified. If the only
change since then is to `CHANGELOG.md` (for example, correcting the date), you
can tag that newer commit instead. Any code or packaging change needs a new
release candidate.

```bash
git tag -s v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

The run pauses at the `pypi` environment. A required reviewer approves it on
the run page, and then the package is published to
https://pypi.org/project/undercurrent/ and the GitHub Release is created
with the `0.1.0` section of `CHANGELOG.md` as its notes.

## 5. Verify the PyPI release

Again, use a fresh virtualenv outside the checkout, with CPU-only torch. Use
a plain `pip install` with no extras and no extra index for `undercurrent`
itself. That's what users run.

```bash
python3.10 -m venv /tmp/uc-final && source /tmp/uc-final/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install undercurrent==0.1.0
python -c "import undercurrent; print(undercurrent.__version__)"   # 0.1.0
cd /tmp && python uc_smoke.py
```

Check that:

- [ ] The smoke test and the README quickstart run.
- [ ] https://pypi.org/project/undercurrent/ shows `0.1.0` as the latest
      version, the README renders, and each file's details on the
      "Download files" page show its provenance attestation.
- [ ] The GitHub Release for `v0.1.0` exists, isn't marked as a
      pre-release, has the changelog notes, and has the wheel and sdist
      attached. Edit the notes on GitHub if they need it.
- [ ] The documentation site (https://wrynx.github.io/undercurrent/) is
      built from `main` and reflects this release. It's deployed from
      `main`, not from tags.

## 6. Post-release

- [ ] Make sure `CHANGELOG.md` on `main` has an empty `## [Unreleased]`
      section at the top and up-to-date compare links. New changes go
      there.

## If something goes wrong

**Never delete a release from PyPI and never reuse a version number.** PyPI
doesn't allow re-uploading a filename even after deletion, and users or
mirrors may already have the files. Fix a bad release by releasing a new
version.

- **A broken final release on PyPI:** [yank](https://pypi.org/help/#yanked)
  it (on PyPI, *Manage project → Releases → Options → Yank*, with a reason).
  pip then ignores it unless someone pins that exact version. Release the
  fix as a new patch version (`0.1.1`) through the normal process. Add a
  note under the yanked version in `CHANGELOG.md` saying it was yanked and
  why, and edit its GitHub Release to say so.
- **A broken release candidate on TestPyPI:** leave it and tag the next
  candidate (`rc2`).
- **The workflow failed before publishing** (build, version check or smoke
  test): nothing was uploaded. Fix the problem on `main` and tag a new
  release candidate. Don't move a tag that has already been pushed.
- **The workflow failed after publishing** (for example in the GitHub
  Release step): don't re-tag. Re-run the failed job from the Actions page,
  or create the GitHub Release by hand from the existing tag.
- **A security issue in a released version:** follow
  [SECURITY.md](SECURITY.md). Fix it privately, release the fix, then
  yank affected versions if they're dangerous to install.

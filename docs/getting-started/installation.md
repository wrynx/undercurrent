# Installation

<!-- owner: p3-quickstart -->

```bash
pip install undercurrent
```

That is all you need on a laptop, in Colab, or anywhere you use Hugging Face
Transformers. The base install includes the Hugging Face backend (`torch` and
`transformers`), so the [Quickstart](quickstart.md) runs with nothing else.

## Which install do I need?

| Where you run it | Command |
| --- | --- |
| Laptop, Colab, Hugging Face Transformers | `pip install undercurrent` |
| An existing vLLM environment or image | `pip install undercurrent`, inside that environment |
| A fresh vLLM setup | `pip install "undercurrent[vllm]"` |

Keep the quotes around `undercurrent[vllm]`: zsh, the default shell on macOS,
treats unquoted square brackets as a glob pattern.

Undercurrent never upgrades your vLLM or torch: it doesn't depend on vLLM at
all, and it only asks for `torch>=2.1`, so it fits whatever versions your vLLM
environment or image already pins. Installing into the environment you
already serve from is the recommended production path. The `[vllm]` extra is a
convenience that installs a vLLM from the tested range. Either way, the vLLM
adapter checks the installed vLLM version when it starts and tells you what
to do if it's outside the supported range.

- [Compatibility](../compatibility.md) lists the supported and tested versions
  of vLLM, torch and transformers.
- [Deploy with vLLM](../guides/vllm-deployment.md) covers installing into a
  vLLM image and serving with probes attached.

## Python versions

Undercurrent supports Python 3.10 and newer.

## CPU-only machines

On Linux, `pip install undercurrent` pulls the default torch wheel, which
bundles CUDA libraries and is several gigabytes. On a machine without a GPU,
install the much smaller CPU-only build of torch from PyTorch's index:

```bash
pip install undercurrent --extra-index-url https://download.pytorch.org/whl/cpu
```

Or install the CPU torch first, then Undercurrent:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install undercurrent
```

On macOS the default torch wheel has no CUDA libraries, so no extra flag is
needed.

## Verify your install

```python
import undercurrent

print(undercurrent.__version__)
```

The console script is installed too:

```bash
undercurrent --version
undercurrent --help
```

## Install from source

To work on Undercurrent itself, clone the repository and install it in
editable mode:

```bash
git clone https://github.com/wrynx/undercurrent.git
cd undercurrent
python3 -m venv .venv && . .venv/bin/activate
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

The development tools (tests, linters, docs build) and the workflow for
contributing are described in [Contributing](../about/contributing.md).

"""Execute the notebooks in notebooks/ top to bottom.

Cells tagged ``skip-ci`` (the ``%pip install`` cell) are dropped: the test
runs against the checkout installed in the current environment. The
notebooks read ``UNDERCURRENT_NB_MODEL`` and default to ``gpt2``; the test
points them at a tiny random GPT-2 so a CPU run takes seconds, not minutes.
``train_probe.ipynb`` also reads its dataset and sample size from the
environment: the test uses the built-in offline ``toy`` dataset (no
``datasets`` download) and the ``examples/`` directory of this checkout.

Run with ``RUN_NETWORK_TESTS=1 pytest tests/notebooks -m slow`` (the tiny model
is downloaded from the Hugging Face Hub on first use).
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS = ["quickstart.ipynb", "train_probe.ipynb"]
SKIP_TAG = "skip-ci"
MODEL_ENV = "UNDERCURRENT_NB_MODEL"
TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"
# Kernel environment for every notebook; notebooks ignore variables they don't read.
NOTEBOOK_ENV = {
    MODEL_ENV: TINY_MODEL,
    "UNDERCURRENT_NB_DATASET": "toy",
    "UNDERCURRENT_NB_N_EXAMPLES": "24",
    "UNDERCURRENT_EXAMPLES_DIR": str(REPO_ROOT / "examples"),
}


def _read(name):
    return nbformat.read(NOTEBOOKS_DIR / name, as_version=4)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_committed_clean(name):
    nb = _read(name)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [], f"{name}: commit the notebook with outputs stripped"
            assert cell.execution_count is None, f"{name}: commit the notebook with execution counts cleared"


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_executes(name, tmp_path, monkeypatch):
    pytest.importorskip("ipykernel")
    nb = _read(name)
    nb.cells = [c for c in nb.cells if SKIP_TAG not in c.metadata.get("tags", [])]
    for key, value in NOTEBOOK_ENV.items():
        monkeypatch.setenv(key, value)  # inherited by the kernel process

    # Files the notebooks write (observations.ndjson, my_probe.safetensors) land in tmp_path.
    client = nbclient.NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    )
    client.execute()

    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors


def test_train_notebook_has_no_probe_hub_api():
    """Probe save/load and Hub publishing are deferred (ROADMAP.md); the notebook uses safetensors directly."""
    nb = _read("train_probe.ipynb")
    text = "\n".join(cell.source for cell in nb.cells)
    for name in ("save_pretrained", "push_to_hub", "torch.load"):
        assert name not in text, name
    assert "save_file" in text and "my_probe.safetensors" in text

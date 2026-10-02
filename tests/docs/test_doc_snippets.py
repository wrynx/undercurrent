"""Execute the ```python blocks in the docs and the README.

The convention (docs/README.md): ```python blocks are complete, runnable,
CPU-only code; blocks on one page run in order and share state. ```py blocks
are illustrative and never run. Each file gets a fresh namespace and runs
with ``tmp_path`` as its working directory, so nothing lands in the repo.

Run with ``RUN_NETWORK_TESTS=1 pytest tests/docs -m slow`` (pages download
``gpt2`` from the Hugging Face Hub on first use).
"""

from pathlib import Path

import pytest

mktestdocs = pytest.importorskip("mktestdocs")

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

MARKDOWN_FILES = [
    *sorted(p for p in DOCS_DIR.rglob("*.md") if "_legacy" not in p.relative_to(DOCS_DIR).parts),
    REPO_ROOT / "README.md",
]


def test_markdown_files_found():
    # Guards against a glob that silently matches nothing.
    assert DOCS_DIR / "index.md" in MARKDOWN_FILES
    assert REPO_ROOT / "README.md" in MARKDOWN_FILES


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_python_snippets_run(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # memory=True concatenates the page's blocks and execs them once in a new
    # namespace, so later blocks see names from earlier ones.
    mktestdocs.check_md_file(path, memory=True, lang="python")

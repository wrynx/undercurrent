# Writing the docs

This directory is the source of the documentation site at
<https://wrynx.github.io/undercurrent/>, built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/). This file is for contributors
and is excluded from the site.

## Build locally

```bash
pip install -e ".[docs]"
mkdocs serve           # live preview at http://127.0.0.1:8000/
mkdocs build --strict  # what CI runs; any warning fails the build
```

## Layout

- The navigation in `mkdocs.yml` is final. Each page lists its owner in a
  comment in the nav and an `<!-- owner: ... -->` comment at the top of the
  page. Fill in your own page; don't add, remove or rename pages.
- `_legacy/` holds the old per-package READMEs. They are raw material for the
  pages and are not part of the site.
- API reference pages use mkdocstrings (`::: undercurrent.<module>`). Docstrings
  use the Google style (`Args:`, `Returns:`, `Raises:`).

## Code blocks

The fence language decides whether CI runs a block:

| Fence | Executed in CI? | Use for |
| --- | --- | --- |
| ` ```python ` | **Yes** | Complete, runnable code |
| ` ```py ` | No | GPU/vLLM-only code and fragments |
| ` ```bash `, ` ```yaml `, ` ```console ` | No | Shell commands, specs, terminal output |

- ` ```python ` blocks run on a **CPU-only** machine. Blocks on the same page
  run in order and share state, so a later block can use names defined in an
  earlier one. Network access is allowed (for Hugging Face Hub downloads such
  as `openai-community/gpt2`). Every block must be complete and runnable: no `...`
  placeholders, no undefined names.
- ` ```py ` blocks are illustrative and never run. Use them for code that needs
  a GPU or vLLM, and for fragments. Put an admonition before GPU-only blocks:

  ```markdown
  !!! info "Requires a GPU"
      This example needs a CUDA GPU and vLLM.
  ```

- ` ```bash `, ` ```yaml ` and ` ```console ` blocks are never executed.

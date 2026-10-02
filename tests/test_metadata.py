"""The installed distribution carries the PyPI metadata from ``pyproject.toml``."""

from importlib.metadata import metadata


def test_license_is_spdx_expression() -> None:
    meta = metadata("undercurrent")
    assert meta["License-Expression"] == "Apache-2.0"
    # PEP 639: a `License ::` classifier must not accompany an SPDX expression.
    assert not any(c.startswith("License ::") for c in meta.get_all("Classifier") or [])


def test_core_metadata_fields() -> None:
    meta = metadata("undercurrent")
    assert meta["Requires-Python"] == ">=3.10"
    assert meta["Summary"].startswith("Undercurrent:")
    assert len(meta["Summary"]) <= 120
    urls = dict(u.split(", ", 1) for u in meta.get_all("Project-URL") or [])
    assert set(urls) >= {"Homepage", "Source", "Issues", "Changelog", "Documentation"}
    assert set(meta.get_all("Provides-Extra") or []) >= {"vllm", "docs", "dev"}

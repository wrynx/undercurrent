"""Invariants of .github/workflows/release.yml.

The trusted publishers configured on PyPI/TestPyPI name this workflow file and
its environments, and the gating decides which index a tag reaches. These
tests stop an innocent-looking edit from breaking either.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"


@pytest.fixture(scope="module")
def wf() -> dict:
    if not WORKFLOW.exists():  # e.g. running from an sdist, which has no .github/
        pytest.skip("release.yml not present")
    return yaml.safe_load(WORKFLOW.read_text())


def test_trigger_is_version_tags_only(wf: dict) -> None:
    # PyYAML parses the bare key `on` as True.
    assert wf[True] == {"push": {"tags": ["v*"]}}


def test_top_level_permissions_read_only_and_no_concurrency(wf: dict) -> None:
    assert wf["permissions"] == {"contents": "read"}
    assert "concurrency" not in wf


def test_environment_names_match_trusted_publishers(wf: dict) -> None:
    jobs = wf["jobs"]
    assert jobs["publish-testpypi"]["environment"]["name"] == "testpypi"
    assert jobs["publish-pypi"]["environment"]["name"] == "pypi"


def test_publish_gating_is_mutually_exclusive(wf: dict) -> None:
    jobs = wf["jobs"]
    assert jobs["publish-testpypi"]["if"] == "needs.build.outputs.prerelease == 'true'"
    assert jobs["publish-pypi"]["if"] == "needs.build.outputs.prerelease == 'false'"
    for name in ("publish-testpypi", "publish-pypi"):
        assert jobs[name]["needs"] == ["build", "smoke"]


def test_write_permissions_only_where_needed(wf: dict) -> None:
    expected = {
        "publish-testpypi": {"id-token": "write"},
        "publish-pypi": {"id-token": "write"},
        "github-release": {"contents": "write"},
    }
    for name, job in wf["jobs"].items():
        assert job.get("permissions") == expected.get(name), name


def test_publishes_with_oidc_not_tokens(wf: dict) -> None:
    text = WORKFLOW.read_text()
    assert "secrets." not in text
    assert "packages:" not in text
    for name, repo_url in (("publish-testpypi", "https://test.pypi.org/legacy/"), ("publish-pypi", None)):
        (step,) = [s for s in wf["jobs"][name]["steps"] if s.get("uses", "").startswith("pypa/gh-action-pypi-publish@")]
        assert step["with"].get("repository-url") == repo_url
        assert step["with"].get("attestations", True) is True
        assert "password" not in step["with"]

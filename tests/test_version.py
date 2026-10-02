"""``undercurrent.__version__`` is derived from git tags by hatch-vcs."""

from packaging.version import Version

import undercurrent


def test_version_is_pep440() -> None:
    assert isinstance(undercurrent.__version__, str)
    assert undercurrent.__version__
    Version(undercurrent.__version__)  # raises InvalidVersion if not PEP 440

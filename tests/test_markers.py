import pytest


@pytest.mark.gpu
def test_gpu_marker_only_runs_with_cuda():
    # tests/conftest.py skips this unless CUDA is usable, so reaching the
    # body means the auto-skip let it through for a real reason.
    import torch

    assert torch.cuda.is_available()


@pytest.mark.network
def test_network_marker_only_runs_when_opted_in():
    # tests/conftest.py skips this unless RUN_NETWORK_TESTS opts in.
    import os

    assert os.environ.get("RUN_NETWORK_TESTS", "").strip().lower() in {"1", "true", "yes"}

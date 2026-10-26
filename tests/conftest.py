import os

import pytest
import torch

from glimmer_slider.upstream import load_upstream


def pytest_addoption(parser):
    parser.addoption("--upstream", default=os.environ.get("PARTICLE_SLIDERS_ROOT"))


@pytest.fixture(scope="session", autouse=True)
def original_api(request):
    root = request.config.getoption("--upstream")
    if root is None:
        raise pytest.UsageError("Supply --upstream=/path/to/pinned/particle-sliders")
    torch.set_num_threads(1)
    return load_upstream(root)

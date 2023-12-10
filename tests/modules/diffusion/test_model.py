import torch

from imaginairy.utils import get_device
from tests.utils import Timer


def test_nonlinearity():
    from torch.nn.functional import silu

    # mps before changes: 1021.54ms
    with Timer("nonlinearity"):
        for _ in range(10):
            for _ in range(11):
                t = torch.randn(1, 512, 64, 64, device=get_device())
                silu(t)
            for _ in range(7):
                t = torch.randn(1, 512, 128, 128, device=get_device())
                silu(t)
            for _ in range(1):
                t = torch.randn(1, 512, 256, 256, device=get_device())
                silu(t)
            for _ in range(5):
                t = torch.randn(1, 256, 256, 256, device=get_device())
                silu(t)
            for _ in range(1):
                t = torch.randn(1, 256, 512, 512, device=get_device())
                silu(t)
            for _ in range(6):
                t = torch.randn(1, 128, 512, 512, device=get_device())
                silu(t)

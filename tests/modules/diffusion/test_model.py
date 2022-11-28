import time

import torch

from imaginairy.modules.diffusion.model import nonlinearity
from imaginairy.utils import get_device


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start

        print(f"{self.name} took {elapsed*1000:.2f} ms")


def test_nonlinearity():
    # mps before changes: 1021.54ms
    with Timer("nonlinearity"):
        for _ in range(10):
            for _ in range(11):
                t = torch.randn(1, 512, 64, 64, device=get_device())
                nonlinearity(t)
            for _ in range(7):
                t = torch.randn(1, 512, 128, 128, device=get_device())
                nonlinearity(t)
            for _ in range(1):
                t = torch.randn(1, 512, 256, 256, device=get_device())
                nonlinearity(t)
            for _ in range(5):
                t = torch.randn(1, 256, 256, 256, device=get_device())
                nonlinearity(t)
            for _ in range(1):
                t = torch.randn(1, 256, 512, 512, device=get_device())
                nonlinearity(t)
            for _ in range(6):
                t = torch.randn(1, 128, 512, 512, device=get_device())
                nonlinearity(t)

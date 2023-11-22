from typing import Any, Tuple

import torch

from imaginairy.modules.sgm.distributions.distributions import (
    DiagonalGaussianDistribution,
)

from .base import AbstractRegularizer


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = {}
        posterior = DiagonalGaussianDistribution(z)
        z = posterior.sample() if self.sample else posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log

from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

import torch

from imaginairy.vendored.refiners.foundationals.latent_diffusion.model import LatentDiffusionModel
from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.ddim import DDIM
from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.scheduler import Scheduler

T = TypeVar("T", bound=LatentDiffusionModel)


def add_noise_interval(
    scheduler: Scheduler,
    /,
    x: torch.Tensor,
    noise: torch.Tensor,
    initial_timestep: torch.Tensor,
    target_timestep: torch.Tensor,
) -> torch.Tensor:
    initial_cumulative_scale_factors = scheduler.cumulative_scale_factors[initial_timestep]
    target_cumulative_scale_factors = scheduler.cumulative_scale_factors[target_timestep]

    factor = target_cumulative_scale_factors / initial_cumulative_scale_factors
    noised_x = factor * x + torch.sqrt(1 - factor**2) * noise
    return noised_x


@dataclass
class Restart(Generic[T]):
    """
    Implements the restart sampling strategy from the paper "Restart Sampling for Improving Generative Processes"
    (https://arxiv.org/pdf/2306.14878.pdf)

    Works only with the DDIM scheduler for now.
    """

    ldm: T
    num_steps: int = 10
    num_iterations: int = 2
    start_time: float = 0.1
    end_time: float = 2

    def __post_init__(self) -> None:
        assert isinstance(self.ldm.scheduler, DDIM), "Restart sampling only works with DDIM scheduler"

    def __call__(
        self,
        x: torch.Tensor,
        /,
        clip_text_embedding: torch.Tensor,
        condition_scale: float = 7.5,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        original_scheduler = self.ldm.scheduler
        new_scheduler = DDIM(self.ldm.scheduler.num_inference_steps, device=self.device, dtype=self.dtype)
        new_scheduler.timesteps = self.timesteps
        self.ldm.scheduler = new_scheduler

        for _ in range(self.num_iterations):
            noise = torch.randn_like(input=x, device=self.device, dtype=self.dtype)
            x = add_noise_interval(
                new_scheduler,
                x=x,
                noise=noise,
                initial_timestep=self.timesteps[-1],
                target_timestep=self.timesteps[0],
            )

            for step in range(len(self.timesteps) - 1):
                x = self.ldm(
                    x, step=step, clip_text_embedding=clip_text_embedding, condition_scale=condition_scale, **kwargs
                )

        self.ldm.scheduler = original_scheduler

        return x

    @cached_property
    def start_step(self) -> int:
        sigmas = self.ldm.scheduler.noise_std / self.ldm.scheduler.cumulative_scale_factors
        return int(torch.argmin(input=torch.abs(input=sigmas[self.ldm.scheduler.timesteps] - self.start_time)))

    @cached_property
    def end_timestep(self) -> int:
        sigmas = self.ldm.scheduler.noise_std / self.ldm.scheduler.cumulative_scale_factors
        return int(torch.argmin(input=torch.abs(input=sigmas - self.end_time)))

    @cached_property
    def timesteps(self) -> torch.Tensor:
        return (
            torch.round(
                torch.linspace(
                    start=int(self.ldm.scheduler.timesteps[self.start_step]),
                    end=self.end_timestep,
                    steps=self.num_steps,
                )
            )
            .flip(0)
            .to(device=self.device, dtype=torch.int64)
        )

    @property
    def device(self) -> torch.device:
        return self.ldm.device

    @property
    def dtype(self) -> torch.dtype:
        return self.ldm.dtype

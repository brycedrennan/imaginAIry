from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from PIL import Image
from torch import Tensor, device as Device, dtype as DType

from imaginairy.vendored.refiners.foundationals.latent_diffusion.model import LatentDiffusionModel

MAX_STEPS = 1000


@dataclass
class DiffusionTarget:
    size: tuple[int, int]
    offset: tuple[int, int]
    clip_text_embedding: Tensor
    init_latents: Tensor | None = None
    mask_latent: Tensor | None = None
    weight: int = 1
    condition_scale: float = 7.5
    start_step: int = 0
    end_step: int = MAX_STEPS

    def crop(self, tensor: Tensor, /) -> Tensor:
        height, width = self.size
        top_offset, left_offset = self.offset
        return tensor[:, :, top_offset : top_offset + height, left_offset : left_offset + width]

    def paste(self, tensor: Tensor, /, crop: Tensor) -> Tensor:
        height, width = self.size
        top_offset, left_offset = self.offset
        tensor[:, :, top_offset : top_offset + height, left_offset : left_offset + width] = crop
        return tensor


T = TypeVar("T", bound=LatentDiffusionModel)
D = TypeVar("D", bound=DiffusionTarget)


@dataclass
class MultiDiffusion(Generic[T, D], ABC):
    ldm: T

    def __call__(self, x: Tensor, /, noise: Tensor, step: int, targets: list[D]) -> Tensor:
        num_updates = torch.zeros_like(input=x)
        cumulative_values = torch.zeros_like(input=x)

        for target in targets:
            match step:
                case step if step == target.start_step and target.init_latents is not None:
                    noise_view = target.crop(noise)
                    view = self.ldm.scheduler.add_noise(
                        x=target.init_latents,
                        noise=noise_view,
                        step=step,
                    )
                case step if target.start_step <= step <= target.end_step:
                    view = target.crop(x)
                case _:
                    continue
            view = self.diffuse_target(x=view, step=step, target=target)
            weight = target.weight * target.mask_latent if target.mask_latent is not None else target.weight
            num_updates = target.paste(num_updates, crop=target.crop(num_updates) + weight)
            cumulative_values = target.paste(cumulative_values, crop=target.crop(cumulative_values) + weight * view)

        return torch.where(condition=num_updates > 0, input=cumulative_values / num_updates, other=x)

    @abstractmethod
    def diffuse_target(self, x: Tensor, step: int, target: D) -> Tensor:
        ...

    @property
    def steps(self) -> list[int]:
        return self.ldm.steps

    @property
    def device(self) -> Device:
        return self.ldm.device

    @property
    def dtype(self) -> DType:
        return self.ldm.dtype

    def decode_latents(self, x: Tensor) -> Image.Image:
        return self.ldm.lda.decode_latents(x=x)

    @staticmethod
    def generate_offset_grid(size: tuple[int, int], stride: int = 8) -> list[tuple[int, int]]:
        height, width = size

        return [
            (y, x)
            for y in range(0, height, stride)
            for x in range(0, width, stride)
            if y + 64 <= height and x + 64 <= width
        ]

from dataclasses import dataclass, field

from PIL import Image
from torch import Tensor

from imaginairy.vendored.refiners.foundationals.latent_diffusion.multi_diffusion import DiffusionTarget, MultiDiffusion
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.model import (
    StableDiffusion_1,
    StableDiffusion_1_Inpainting,
)


class SD1MultiDiffusion(MultiDiffusion[StableDiffusion_1, DiffusionTarget]):
    def diffuse_target(self, x: Tensor, step: int, target: DiffusionTarget) -> Tensor:
        return self.ldm(
            x=x,
            step=step,
            clip_text_embedding=target.clip_text_embedding,
            scale=target.condition_scale,
        )


@dataclass
class InpaintingDiffusionTarget(DiffusionTarget):
    target_image: Image.Image = field(default_factory=lambda: Image.new(mode="RGB", size=(512, 512), color=255))
    mask: Image.Image = field(default_factory=lambda: Image.new(mode="L", size=(512, 512), color=255))


class SD1InpaintingMultiDiffusion(MultiDiffusion[StableDiffusion_1_Inpainting, InpaintingDiffusionTarget]):
    def diffuse_target(self, x: Tensor, step: int, target: InpaintingDiffusionTarget) -> Tensor:
        self.ldm.set_inpainting_conditions(
            target_image=target.target_image,
            mask=target.mask,
        )

        return self.ldm(
            x=x,
            step=step,
            clip_text_embedding=target.clip_text_embedding,
            scale=target.condition_scale,
        )

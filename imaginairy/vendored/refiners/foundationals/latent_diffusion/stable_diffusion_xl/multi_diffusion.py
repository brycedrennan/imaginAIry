from torch import Tensor

from imaginairy.vendored.refiners.foundationals.latent_diffusion.multi_diffusion import DiffusionTarget, MultiDiffusion
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.model import StableDiffusion_XL


class SDXLDiffusionTarget(DiffusionTarget):
    pooled_text_embedding: Tensor
    time_ids: Tensor


class SDXLMultiDiffusion(MultiDiffusion[StableDiffusion_XL, SDXLDiffusionTarget]):
    def diffuse_target(self, x: Tensor, step: int, target: SDXLDiffusionTarget) -> Tensor:
        return self.ldm(
            x=x,
            step=step,
            clip_text_embedding=target.clip_text_embedding,
            pooled_text_embedding=target.pooled_text_embedding,
            time_ids=target.time_ids,
            condition_scale=target.condition_scale,
        )

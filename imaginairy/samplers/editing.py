"""
Wrapper for instruct pix2pix model.

modified from https://github.com/timothybrooks/instruct-pix2pix/blob/main/edit_cli.py
"""
import torch
from einops import einops
from torch import nn

from imaginairy.samplers.base import mask_blend


class CFGEditingDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(
        self,
        z,
        sigma,
        cond,
        uncond,
        cond_scale,
        image_cfg_scale=1.5,
        mask=None,
        mask_noise=None,
        orig_latent=None,
    ):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [
                torch.cat(
                    [
                        cond["c_crossattn"][0],
                        uncond["c_crossattn"][0],
                        uncond["c_crossattn"][0],
                    ]
                )
            ],
            "c_concat": [
                torch.cat(
                    [cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]]
                )
            ],
        }

        if mask is not None:
            assert orig_latent is not None
            t = self.inner_model.sigma_to_t(sigma, quantize=True)
            big_sigma = max(sigma, 1)
            cfg_z = mask_blend(
                noisy_latent=cfg_z,
                orig_latent=orig_latent * big_sigma,
                mask=mask,
                mask_noise=mask_noise * big_sigma,
                ts=t,
                model=self.inner_model.inner_model,
            )

        out_cond, out_img_cond, out_uncond = self.inner_model(
            cfg_z, cfg_sigma, cond=cfg_cond
        ).chunk(3)

        result = (
            out_uncond
            + cond_scale * (out_cond - out_img_cond)
            + image_cfg_scale * (out_img_cond - out_uncond)
        )

        return result

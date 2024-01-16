from typing import Literal

import torch
import torch.nn as nn
from safetensors.torch import load_file

from imaginairy.utils.downloads import get_cached_url_path

LatentVerType = Literal["v1", "xl"]
ScaleFactorType = Literal["1.25", "1.5", "2.0"]


class Upscaler(nn.Module):
    """
    Basic NN layout, ported from:
    https://github.com/city96/SD-Latent-Upscaler/blob/main/upscaler.py
    """

    version = 2.1  # network revision

    def head(self):
        return [
            nn.Conv2d(self.chan, self.size, kernel_size=self.krn, padding=self.pad),
            nn.ReLU(),
            nn.Upsample(scale_factor=self.fac, mode="nearest"),
            nn.ReLU(),
        ]

    def core(self):
        layers = []
        for _ in range(self.depth):
            layers += [
                nn.Conv2d(self.size, self.size, kernel_size=self.krn, padding=self.pad),
                nn.ReLU(),
            ]
        return layers

    def tail(self):
        return [
            nn.Conv2d(self.size, self.chan, kernel_size=self.krn, padding=self.pad),
        ]

    def __init__(self, fac, depth=16):
        super().__init__()
        self.size = 64  # Conv2d size
        self.chan = 4  # in/out channels
        self.depth = depth  # no. of layers
        self.fac = fac  # scale factor
        self.krn = 3  # kernel size
        self.pad = 1  # padding

        self.sequential = nn.Sequential(
            *self.head(),
            *self.core(),
            *self.tail(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


def upscale_latent(
    latent: torch.Tensor, latent_ver: LatentVerType, scale_factor: ScaleFactorType
):
    model = Upscaler(scale_factor)
    orig_dtype, orig_device = latent.dtype, latent.device
    latent = latent.to(dtype=torch.float32, device="cpu")
    latent = latent / 0.13025
    filename = (
        f"latent-upscaler-v{model.version}_SD{latent_ver}-x{scale_factor}.safetensors"
    )
    weights_url = f"https://huggingface.co/city96/SD-Latent-Upscaler/resolve/99c65021fa947dfe3d71ec4e24793fe7533a3322/{filename}"
    weights_path = get_cached_url_path(weights_url)

    model.load_state_dict(load_file(weights_path), assign=True)

    big_latent = model(latent)
    big_latent = big_latent.to(dtype=orig_dtype, device=orig_device)
    del model
    big_latent = big_latent * 0.13025
    return big_latent

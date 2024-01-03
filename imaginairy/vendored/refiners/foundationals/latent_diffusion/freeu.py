import math
from typing import Any, Callable, Generic, TypeVar

import torch
from torch import Tensor
from torch.fft import fftn, fftshift, ifftn, ifftshift  # type: ignore

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import ResidualConcatenator, SD1UNet
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TSDFreeUAdapter = TypeVar("TSDFreeUAdapter", bound="SDFreeUAdapter[Any]")  # Self (see PEP 673)


def fourier_filter(x: Tensor, scale: float = 1, threshold: int = 1) -> Tensor:
    """Fourier filter as introduced in FreeU (https://arxiv.org/abs/2309.11497).

    This version of the method comes from here:
    https://github.com/ChenyangSi/FreeU/blob/main/demo/free_lunch_utils.py#L23
    """
    batch, channels, height, width = x.shape
    dtype = x.dtype
    device = x.device

    if not (math.log2(height).is_integer() and math.log2(width).is_integer()):
        x = x.to(dtype=torch.float32)

    x_freq = fftn(x, dim=(-2, -1))  # type: ignore
    x_freq = fftshift(x_freq, dim=(-2, -1))  # type: ignore
    mask = torch.ones((batch, channels, height, width), device=device)  # type: ignore

    center_row, center_col = height // 2, width // 2  # type: ignore
    mask[..., center_row - threshold : center_row + threshold, center_col - threshold : center_col + threshold] = scale
    x_freq = x_freq * mask  # type: ignore

    x_freq = ifftshift(x_freq, dim=(-2, -1))  # type: ignore
    x_filtered = ifftn(x_freq, dim=(-2, -1)).real  # type: ignore

    return x_filtered.to(dtype=dtype)  # type: ignore


class FreeUBackboneFeatures(fl.Module):
    def __init__(self, backbone_scale: float) -> None:
        super().__init__()
        self.backbone_scale = backbone_scale

    def forward(self, x: Tensor) -> Tensor:
        num_half_channels = x.shape[1] // 2
        x[:, :num_half_channels] = x[:, :num_half_channels] * self.backbone_scale
        return x


class FreeUSkipFeatures(fl.Chain):
    def __init__(self, n: int, skip_scale: float) -> None:
        apply_filter: Callable[[Tensor], Tensor] = lambda x: fourier_filter(x, scale=skip_scale)
        super().__init__(
            fl.UseContext(context="unet", key="residuals").compose(lambda residuals: residuals[n]),
            fl.Lambda(apply_filter),
        )


class FreeUResidualConcatenator(fl.Concatenate):
    def __init__(self, n: int, backbone_scale: float, skip_scale: float) -> None:
        super().__init__(
            FreeUBackboneFeatures(backbone_scale),
            FreeUSkipFeatures(n, skip_scale),
            dim=1,
        )


class SDFreeUAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(self, target: T, backbone_scales: list[float], skip_scales: list[float]) -> None:
        assert len(backbone_scales) == len(skip_scales)
        assert len(backbone_scales) <= len(target.UpBlocks)
        self.backbone_scales = backbone_scales
        self.skip_scales = skip_scales
        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self: TSDFreeUAdapter, parent: fl.Chain | None = None) -> TSDFreeUAdapter:
        for n, (backbone_scale, skip_scale) in enumerate(zip(self.backbone_scales, self.skip_scales)):
            block = self.target.UpBlocks[n]
            concat = block.ensure_find(ResidualConcatenator)
            block.replace(concat, FreeUResidualConcatenator(-n - 2, backbone_scale, skip_scale))
        return super().inject(parent)

    def eject(self) -> None:
        for n in range(len(self.backbone_scales)):
            block = self.target.UpBlocks[n]
            concat = block.ensure_find(FreeUResidualConcatenator)
            block.replace(concat, ResidualConcatenator(-n - 2))
        super().eject()

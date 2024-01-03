from typing import Iterable, cast

from torch import Tensor, device as Device, dtype as DType

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from imaginairy.vendored.refiners.foundationals.latent_diffusion.range_adapter import RangeAdapter2d, RangeEncoder


class TimestepEncoder(fl.Passthrough):
    def __init__(
        self,
        context_key: str = "timestep_embedding",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.UseContext("diffusion", "timestep"),
            RangeEncoder(320, 1280, device=device, dtype=dtype),
            fl.SetContext("range_adapter", context_key),
        )


class ResidualBlock(fl.Sum):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        if in_channels % num_groups != 0 or out_channels % num_groups != 0:
            raise ValueError("Number of input and output channels must be divisible by num_groups.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.eps = eps
        shortcut = (
            fl.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device, dtype=dtype)
            if in_channels != out_channels
            else fl.Identity()
        )
        super().__init__(
            fl.Chain(
                fl.GroupNorm(channels=in_channels, num_groups=num_groups, eps=eps, device=device, dtype=dtype),
                fl.SiLU(),
                fl.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                fl.GroupNorm(channels=out_channels, num_groups=num_groups, eps=eps, device=device, dtype=dtype),
                fl.SiLU(),
                fl.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
            shortcut,
        )


class CLIPLCrossAttention(CrossAttentionBlock2d):
    def __init__(
        self,
        channels: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            channels=channels,
            context_embedding_dim=768,
            context_key="clip_text_embedding",
            num_attention_heads=8,
            use_bias=False,
            device=device,
            dtype=dtype,
        )


class DownBlocks(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.in_channels = in_channels
        super().__init__(
            fl.Chain(
                fl.Conv2d(
                    in_channels=in_channels, out_channels=320, kernel_size=3, padding=1, device=device, dtype=dtype
                )
            ),
            fl.Chain(
                ResidualBlock(in_channels=320, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=320, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(fl.Downsample(channels=320, scale_factor=2, padding=1, device=device, dtype=dtype)),
            fl.Chain(
                ResidualBlock(in_channels=320, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(fl.Downsample(channels=640, scale_factor=2, padding=1, device=device, dtype=dtype)),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(fl.Downsample(channels=1280, scale_factor=2, padding=1, device=device, dtype=dtype)),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
            ),
        )


class UpBlocks(fl.Chain):
    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
                fl.Upsample(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1920, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
                fl.Upsample(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1920, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=960, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
                fl.Upsample(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=960, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
        )


class MiddleBlock(fl.Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
            CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
        )


class ResidualAccumulator(fl.Passthrough):
    def __init__(self, n: int) -> None:
        self.n = n

        super().__init__(
            fl.Residual(
                fl.UseContext(context="unet", key="residuals").compose(func=lambda residuals: residuals[self.n])
            ),
            fl.SetContext(context="unet", key="residuals", callback=self.update),
        )

    def update(self, residuals: list[Tensor | float], x: Tensor) -> None:
        residuals[self.n] = x


class ResidualConcatenator(fl.Chain):
    def __init__(self, n: int) -> None:
        self.n = n

        super().__init__(
            fl.Concatenate(
                fl.Identity(),
                fl.UseContext(context="unet", key="residuals").compose(lambda residuals: residuals[self.n]),
                dim=1,
            ),
        )


class SD1UNet(fl.Chain):
    def __init__(self, in_channels: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.in_channels = in_channels
        super().__init__(
            TimestepEncoder(device=device, dtype=dtype),
            DownBlocks(in_channels=in_channels, device=device, dtype=dtype),
            fl.Sum(
                fl.UseContext(context="unet", key="residuals").compose(lambda x: x[-1]),
                MiddleBlock(device=device, dtype=dtype),
            ),
            UpBlocks(),
            fl.Chain(
                fl.GroupNorm(channels=320, num_groups=32, device=device, dtype=dtype),
                fl.SiLU(),
                fl.Conv2d(
                    in_channels=320,
                    out_channels=4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )
        for residual_block in self.layers(ResidualBlock):
            chain = residual_block.Chain
            RangeAdapter2d(
                target=chain.Conv2d_1,
                channels=residual_block.out_channels,
                embedding_dim=1280,
                context_key="timestep_embedding",
                device=device,
                dtype=dtype,
            ).inject(chain)
        for n, block in enumerate(cast(Iterable[fl.Chain], self.DownBlocks)):
            block.append(ResidualAccumulator(n))
        for n, block in enumerate(cast(Iterable[fl.Chain], self.UpBlocks)):
            block.insert(0, ResidualConcatenator(-n - 2))

    def init_context(self) -> Contexts:
        return {
            "unet": {"residuals": [0.0] * 13},
            "diffusion": {"timestep": None},
            "range_adapter": {"timestep_embedding": None},
            "sampling": {"shapes": []},
        }

    def set_clip_text_embedding(self, clip_text_embedding: Tensor) -> None:
        self.set_context("cross_attention_block", {"clip_text_embedding": clip_text_embedding})

    def set_timestep(self, timestep: Tensor) -> None:
        self.set_context("diffusion", {"timestep": timestep})

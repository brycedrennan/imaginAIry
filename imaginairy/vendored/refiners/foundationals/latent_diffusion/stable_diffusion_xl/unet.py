from typing import cast

from torch import Tensor, device as Device, dtype as DType

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from imaginairy.vendored.refiners.foundationals.latent_diffusion.range_adapter import (
    RangeAdapter2d,
    RangeEncoder,
    compute_sinusoidal_embedding,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import (
    ResidualAccumulator,
    ResidualBlock,
    ResidualConcatenator,
)


class TextTimeEmbedding(fl.Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.timestep_embedding_dim = 1280
        self.time_ids_embedding_dim = 256
        self.text_time_embedding_dim = 2816
        super().__init__(
            fl.Concatenate(
                fl.UseContext(context="diffusion", key="pooled_text_embedding"),
                fl.Chain(
                    fl.UseContext(context="diffusion", key="time_ids"),
                    fl.Unsqueeze(dim=-1),
                    fl.Lambda(func=self.compute_sinusoidal_embedding),
                    fl.Reshape(-1),
                ),
                dim=1,
            ),
            fl.Converter(set_device=False, set_dtype=True),
            fl.Linear(
                in_features=self.text_time_embedding_dim,
                out_features=self.timestep_embedding_dim,
                device=device,
                dtype=dtype,
            ),
            fl.SiLU(),
            fl.Linear(
                in_features=self.timestep_embedding_dim,
                out_features=self.timestep_embedding_dim,
                device=device,
                dtype=dtype,
            ),
        )

    def compute_sinusoidal_embedding(self, x: Tensor) -> Tensor:
        return compute_sinusoidal_embedding(x=x, embedding_dim=self.time_ids_embedding_dim)


class TimestepEncoder(fl.Passthrough):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.timestep_embedding_dim = 1280
        super().__init__(
            fl.Sum(
                fl.Chain(
                    fl.UseContext(context="diffusion", key="timestep"),
                    RangeEncoder(
                        sinusoidal_embedding_dim=320,
                        embedding_dim=self.timestep_embedding_dim,
                        device=device,
                        dtype=dtype,
                    ),
                ),
                TextTimeEmbedding(device=device, dtype=dtype),
            ),
            fl.SetContext(context="range_adapter", key="timestep_embedding"),
        )


class SDXLCrossAttention(CrossAttentionBlock2d):
    def __init__(
        self,
        channels: int,
        num_attention_layers: int = 1,
        num_attention_heads: int = 10,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            channels=channels,
            context_embedding_dim=2048,
            context_key="clip_text_embedding",
            num_attention_layers=num_attention_layers,
            num_attention_heads=num_attention_heads,
            use_bias=False,
            use_linear_projection=True,
            device=device,
            dtype=dtype,
        )


class DownBlocks(fl.Chain):
    def __init__(self, in_channels: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.in_channels = in_channels

        in_block = fl.Chain(
            fl.Conv2d(in_channels=in_channels, out_channels=320, kernel_size=3, padding=1, device=device, dtype=dtype)
        )
        first_blocks = [
            fl.Chain(
                ResidualBlock(in_channels=320, out_channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=320, out_channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                fl.Downsample(channels=320, scale_factor=2, padding=1, device=device, dtype=dtype),
            ),
        ]
        second_blocks = [
            fl.Chain(
                ResidualBlock(in_channels=320, out_channels=640, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=640, num_attention_layers=2, num_attention_heads=10, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=640, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=640, num_attention_layers=2, num_attention_heads=10, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                fl.Downsample(channels=640, scale_factor=2, padding=1, device=device, dtype=dtype),
            ),
        ]
        third_blocks = [
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=1280, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=1280, num_attention_layers=10, num_attention_heads=20, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=1280, num_attention_layers=10, num_attention_heads=20, device=device, dtype=dtype
                ),
            ),
        ]

        super().__init__(
            in_block,
            *first_blocks,
            *second_blocks,
            *third_blocks,
        )


class UpBlocks(fl.Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        first_blocks = [
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=1280, num_attention_layers=10, num_attention_heads=20, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=1280, num_attention_layers=10, num_attention_heads=20, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1920, out_channels=1280, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=1280, num_attention_layers=10, num_attention_heads=20, device=device, dtype=dtype
                ),
                fl.Upsample(channels=1280, device=device, dtype=dtype),
            ),
        ]

        second_blocks = [
            fl.Chain(
                ResidualBlock(in_channels=1920, out_channels=640, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=640, num_attention_layers=2, num_attention_heads=10, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=640, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=640, num_attention_layers=2, num_attention_heads=10, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(in_channels=960, out_channels=640, device=device, dtype=dtype),
                SDXLCrossAttention(
                    channels=640, num_attention_layers=2, num_attention_heads=10, device=device, dtype=dtype
                ),
                fl.Upsample(channels=640, device=device, dtype=dtype),
            ),
        ]

        third_blocks = [
            fl.Chain(
                ResidualBlock(in_channels=960, out_channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=320, device=device, dtype=dtype),
            ),
        ]

        super().__init__(
            *first_blocks,
            *second_blocks,
            *third_blocks,
        )


class MiddleBlock(fl.Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
            SDXLCrossAttention(
                channels=1280, num_attention_layers=10, num_attention_heads=20, device=device, dtype=dtype
            ),
            ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
        )


class OutputBlock(fl.Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            fl.GroupNorm(channels=320, num_groups=32, device=device, dtype=dtype),
            fl.SiLU(),
            fl.Conv2d(in_channels=320, out_channels=4, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype),
        )


class SDXLUNet(fl.Chain):
    def __init__(self, in_channels: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.in_channels = in_channels
        super().__init__(
            TimestepEncoder(device=device, dtype=dtype),
            DownBlocks(in_channels=in_channels, device=device, dtype=dtype),
            MiddleBlock(device=device, dtype=dtype),
            fl.Residual(fl.UseContext(context="unet", key="residuals").compose(lambda x: x[-1])),
            UpBlocks(device=device, dtype=dtype),
            OutputBlock(device=device, dtype=dtype),
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
        for n, block in enumerate(iterable=cast(list[fl.Chain], self.DownBlocks)):
            block.append(module=ResidualAccumulator(n=n))
        for n, block in enumerate(iterable=cast(list[fl.Chain], self.UpBlocks)):
            block.insert(index=0, module=ResidualConcatenator(n=-n - 2))

    def init_context(self) -> Contexts:
        return {
            "unet": {"residuals": [0.0] * 10},
            "diffusion": {"timestep": None, "time_ids": None, "pooled_text_embedding": None},
            "range_adapter": {"timestep_embedding": None},
            "sampling": {"shapes": []},
        }

    def set_clip_text_embedding(self, clip_text_embedding: Tensor) -> None:
        self.set_context(context="cross_attention_block", value={"clip_text_embedding": clip_text_embedding})

    def set_timestep(self, timestep: Tensor) -> None:
        self.set_context(context="diffusion", value={"timestep": timestep})

    def set_time_ids(self, time_ids: Tensor) -> None:
        self.set_context(context="diffusion", value={"time_ids": time_ids})

    def set_pooled_text_embedding(self, pooled_text_embedding: Tensor) -> None:
        self.set_context(context="diffusion", value={"pooled_text_embedding": pooled_text_embedding})

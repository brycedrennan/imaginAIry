from typing import TYPE_CHECKING, Any, Generic, TypeVar

from torch import Tensor, device as Device, dtype as DType
from torch.nn import AvgPool2d as _AvgPool2d

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter
from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.fluxion.layers.module import Module

if TYPE_CHECKING:
    from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
    from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TT2IAdapter = TypeVar("TT2IAdapter", bound="T2IAdapter[Any]")  # Self (see PEP 673)


class Downsample2d(_AvgPool2d, Module):
    def __init__(self, scale_factor: int) -> None:
        _AvgPool2d.__init__(self, kernel_size=scale_factor, stride=scale_factor)


class ResidualBlock(fl.Residual):
    def __init__(
        self,
        channels: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Conv2d(
                in_channels=channels, out_channels=channels, kernel_size=3, padding=1, device=device, dtype=dtype
            ),
            fl.ReLU(),
            fl.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, device=device, dtype=dtype),
        )


class ResidualBlocks(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_residual_blocks: int = 2,
        downsample: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        preproc = Downsample2d(scale_factor=2) if downsample else fl.Identity()
        shortcut = (
            fl.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device, dtype=dtype)
            if in_channels != out_channels
            else fl.Identity()
        )
        super().__init__(
            preproc,
            shortcut,
            fl.Chain(
                ResidualBlock(channels=out_channels, device=device, dtype=dtype) for _ in range(num_residual_blocks)
            ),
        )


class StatefulResidualBlocks(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_residual_blocks: int = 2,
        downsample: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            ResidualBlocks(
                in_channels=in_channels,
                out_channels=out_channels,
                num_residual_blocks=num_residual_blocks,
                downsample=downsample,
                device=device,
                dtype=dtype,
            ),
            fl.SetContext(context="t2iadapter", key="features", callback=self.push),
        )

    def push(self, features: list[Tensor], x: Tensor) -> None:
        features.append(x)


class ConditionEncoder(fl.Chain):
    def __init__(
        self,
        in_channels: int = 3,
        channels: tuple[int, int, int, int] = (320, 640, 1280, 1280),
        num_residual_blocks: int = 2,
        downscale_factor: int = 8,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.scale = scale
        super().__init__(
            fl.PixelUnshuffle(downscale_factor=downscale_factor),
            fl.Conv2d(
                in_channels=in_channels * downscale_factor**2,
                out_channels=channels[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            StatefulResidualBlocks(channels[0], channels[0], num_residual_blocks, device=device, dtype=dtype),
            *(
                StatefulResidualBlocks(
                    channels[i - 1], channels[i], num_residual_blocks, downsample=True, device=device, dtype=dtype
                )
                for i in range(1, len(channels))
            ),
            fl.UseContext(context="t2iadapter", key="features"),
        )

    def init_context(self) -> Contexts:
        return {"t2iadapter": {"features": []}}


class ConditionEncoderXL(ConditionEncoder, fl.Chain):
    def __init__(
        self,
        in_channels: int = 3,
        channels: tuple[int, int, int, int] = (320, 640, 1280, 1280),
        num_residual_blocks: int = 2,
        downscale_factor: int = 16,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.scale = scale
        fl.Chain.__init__(
            self,
            fl.PixelUnshuffle(downscale_factor=downscale_factor),
            fl.Conv2d(
                in_channels=in_channels * downscale_factor**2,
                out_channels=channels[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            StatefulResidualBlocks(channels[0], channels[0], num_residual_blocks, device=device, dtype=dtype),
            StatefulResidualBlocks(channels[0], channels[1], num_residual_blocks, device=device, dtype=dtype),
            StatefulResidualBlocks(
                channels[1], channels[2], num_residual_blocks, downsample=True, device=device, dtype=dtype
            ),
            StatefulResidualBlocks(channels[2], channels[3], num_residual_blocks, device=device, dtype=dtype),
            fl.UseContext(context="t2iadapter", key="features"),
        )


class T2IFeatures(fl.Residual):
    def __init__(self, name: str, index: int, scale: float = 1.0) -> None:
        self.name = name
        self.index = index
        self.scale = scale
        super().__init__(
            fl.UseContext(context="t2iadapter", key=f"condition_features_{self.name}").compose(
                func=lambda features: self.scale * features[self.index]
            )
        )


class T2IAdapter(Generic[T], fl.Chain, Adapter[T]):
    _condition_encoder: list[ConditionEncoder]  # prevent PyTorch module registration
    _features: list[T2IFeatures] = []

    def __init__(
        self,
        target: T,
        name: str,
        condition_encoder: ConditionEncoder,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        self.name = name
        if weights is not None:
            condition_encoder.load_state_dict(weights)
        self._condition_encoder = [condition_encoder]

        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self: TT2IAdapter, parent: fl.Chain | None = None) -> TT2IAdapter:
        return super().inject(parent)

    def eject(self) -> None:
        super().eject()

    @property
    def condition_encoder(self) -> ConditionEncoder:
        return self._condition_encoder[0]

    def compute_condition_features(self, condition: Tensor) -> tuple[Tensor, ...]:
        return self.condition_encoder(condition)

    def set_condition_features(self, features: tuple[Tensor, ...]) -> None:
        self.set_context("t2iadapter", {f"condition_features_{self.name}": features})

    def set_scale(self, scale: float) -> None:
        for f in self._features:
            f.scale = scale

    def init_context(self) -> Contexts:
        return {"t2iadapter": {f"condition_features_{self.name}": None}}

    def structural_copy(self: "TT2IAdapter") -> "TT2IAdapter":
        raise RuntimeError("T2I-Adapter cannot be copied, eject it first.")

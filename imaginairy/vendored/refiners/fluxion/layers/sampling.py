from typing import Callable

from torch import Size, Tensor, device as Device, dtype as DType
from torch.nn.functional import pad

from imaginairy.vendored.refiners.fluxion.layers.basics import Identity
from imaginairy.vendored.refiners.fluxion.layers.chain import Chain, Lambda, Parallel, SetContext, UseContext
from imaginairy.vendored.refiners.fluxion.layers.conv import Conv2d
from imaginairy.vendored.refiners.fluxion.layers.module import Module
from imaginairy.vendored.refiners.fluxion.utils import interpolate


class Downsample(Chain):
    def __init__(
        self,
        channels: int,
        scale_factor: int,
        padding: int = 0,
        register_shape: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        """Downsamples the input by the given scale factor.

        If register_shape is True, the input shape is registered in the context. It will throw an error if the context
        sampling is not set or if the context does not contain a list.
        """
        self.channels = channels
        self.in_channels = channels
        self.out_channels = channels
        self.scale_factor = scale_factor
        self.padding = padding
        super().__init__(
            Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=scale_factor,
                padding=padding,
                device=device,
                dtype=dtype,
            ),
        )
        if padding == 0:
            zero_pad: Callable[[Tensor], Tensor] = lambda x: pad(x, (0, 1, 0, 1))
            self.insert(0, Lambda(zero_pad))
        if register_shape:
            self.insert(0, SetContext(context="sampling", key="shapes", callback=self.register_shape))

    def register_shape(self, shapes: list[Size], x: Tensor) -> None:
        shapes.append(x.shape[2:])


class Interpolate(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, shape: Size) -> Tensor:
        return interpolate(x, shape)


class Upsample(Chain):
    def __init__(
        self,
        channels: int,
        upsample_factor: int | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        """Upsamples the input by the given scale factor.

        If upsample_factor is None, the input shape is taken from the context. It will throw an error if the context
        sampling is not set or if the context is empty (then you should use the dynamic version of Downsample).
        """
        self.channels = channels
        self.upsample_factor = upsample_factor
        super().__init__(
            Parallel(
                Identity(),
                (
                    Lambda(self._get_static_shape)
                    if upsample_factor is not None
                    else UseContext(context="sampling", key="shapes").compose(lambda x: x.pop())
                ),
            ),
            Interpolate(),
            Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
        )

    def _get_static_shape(self, x: Tensor) -> Size:
        assert self.upsample_factor is not None
        return Size([size * self.upsample_factor for size in x.shape[2:]])

from torch import device as Device, dtype as DType, nn

from imaginairy.vendored.refiners.fluxion.layers.module import WeightedModule


class Conv2d(nn.Conv2d, WeightedModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = (1, 1),
        padding: int | tuple[int, int] | str = (0, 0),
        groups: int = 1,
        use_bias: bool = True,
        dilation: int | tuple[int, int] = (1, 1),
        padding_mode: str = "zeros",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            device,
            dtype,
        )
        self.use_bias = use_bias


class Conv1d(nn.Conv1d, WeightedModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        groups: int = 1,
        use_bias: bool = True,
        dilation: int | tuple[int] = 1,
        padding_mode: str = "zeros",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            device,
            dtype,
        )


class ConvTranspose2d(nn.ConvTranspose2d, WeightedModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        use_bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        padding_mode: str = "zeros",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

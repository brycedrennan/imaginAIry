from torch import nn

from imaginairy.vendored.refiners.fluxion.layers.module import Module


class MaxPool1d(nn.MaxPool1d, Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )


class MaxPool2d(nn.MaxPool2d, Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = (0, 0),
        dilation: int | tuple[int, int] = (1, 1),
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,  # type: ignore
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

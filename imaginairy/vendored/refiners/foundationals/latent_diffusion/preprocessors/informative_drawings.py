# Adapted from https://github.com/carolineec/informative-drawings, MIT License

from torch import device as Device, dtype as DType

import imaginairy.vendored.refiners.fluxion.layers as fl


class InformativeDrawings(fl.Chain):
    """Model typically used as the preprocessor for the Lineart ControlNet.

    Implements the paper "Learning to generate line drawings that convey
    geometry and semantics" published in 2022 by Caroline Chan, Frédo Durand
    and Phillip Isola - https://arxiv.org/abs/2203.12691

    For use as a preprocessor it is recommended to use the weights for "Style 2".
    """

    def __init__(
        self,
        in_channels: int = 3,  # RGB
        out_channels: int = 1,  # Grayscale
        n_residual_blocks: int = 3,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Chain(  # Initial convolution
                fl.ReflectionPad2d(3),
                fl.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=7,
                    device=device,
                    dtype=dtype,
                ),
                fl.InstanceNorm2d(64, device=device, dtype=dtype),
                fl.ReLU(),
            ),
            *(  # Downsampling
                fl.Chain(
                    fl.Conv2d(
                        in_channels=64 * (2**i),
                        out_channels=128 * (2**i),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        device=device,
                        dtype=dtype,
                    ),
                    fl.InstanceNorm2d(128 * (2**i), device=device, dtype=dtype),
                    fl.ReLU(),
                )
                for i in range(2)
            ),
            *(  # Residual blocks
                fl.Residual(
                    fl.ReflectionPad2d(1),
                    fl.Conv2d(
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        device=device,
                        dtype=dtype,
                    ),
                    fl.InstanceNorm2d(256, device=device, dtype=dtype),
                    fl.ReLU(),
                    fl.ReflectionPad2d(1),
                    fl.Conv2d(
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        device=device,
                        dtype=dtype,
                    ),
                    fl.InstanceNorm2d(256, device=device, dtype=dtype),
                )
                for _ in range(n_residual_blocks)
            ),
            *(  # Upsampling
                fl.Chain(
                    fl.ConvTranspose2d(
                        in_channels=128 * (2**i),
                        out_channels=64 * (2**i),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        device=device,
                        dtype=dtype,
                    ),
                    fl.InstanceNorm2d(64 * (2**i), device=device, dtype=dtype),
                    fl.ReLU(),
                )
                for i in reversed(range(2))
            ),
            fl.Chain(  # Output layer
                fl.ReflectionPad2d(3),
                fl.Conv2d(
                    in_channels=64,
                    out_channels=out_channels,
                    kernel_size=7,
                    device=device,
                    dtype=dtype,
                ),
                fl.Sigmoid(),
            ),
        )

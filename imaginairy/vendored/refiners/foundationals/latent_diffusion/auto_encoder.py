from PIL import Image
from torch import Tensor, device as Device, dtype as DType

from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.fluxion.layers import (
    Chain,
    Conv2d,
    Downsample,
    GroupNorm,
    Identity,
    Residual,
    SelfAttention2d,
    SiLU,
    Slicing,
    Sum,
    Upsample,
)
from imaginairy.vendored.refiners.fluxion.utils import image_to_tensor, tensor_to_image


class Resnet(Sum):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        shortcut = (
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device, dtype=dtype)
            if in_channels != out_channels
            else Identity()
        )
        super().__init__(
            shortcut,
            Chain(
                GroupNorm(channels=in_channels, num_groups=num_groups, device=device, dtype=dtype),
                SiLU(),
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                GroupNorm(channels=out_channels, num_groups=num_groups, device=device, dtype=dtype),
                SiLU(),
                Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class Encoder(Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        resnet_sizes: list[int] = [128, 256, 512, 512, 512]
        input_channels: int = 3
        latent_dim: int = 8
        resnet_layers: list[Chain] = [
            Chain(
                [
                    Resnet(
                        in_channels=resnet_sizes[i - 1] if i > 0 else resnet_sizes[0],
                        out_channels=resnet_sizes[i],
                        device=device,
                        dtype=dtype,
                    ),
                    Resnet(in_channels=resnet_sizes[i], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                ]
            )
            for i in range(len(resnet_sizes))
        ]
        for _, layer in zip(range(3), resnet_layers):
            channels: int = layer[-1].out_channels  # type: ignore
            layer.append(Downsample(channels=channels, scale_factor=2, device=device, dtype=dtype))

        attention_layer = Residual(
            GroupNorm(channels=resnet_sizes[-1], num_groups=32, eps=1e-6, device=device, dtype=dtype),
            SelfAttention2d(channels=resnet_sizes[-1], device=device, dtype=dtype),
        )
        resnet_layers[-1].insert_after_type(Resnet, attention_layer)
        super().__init__(
            Conv2d(
                in_channels=input_channels,
                out_channels=resnet_sizes[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            Chain(*resnet_layers),
            Chain(
                GroupNorm(channels=resnet_sizes[-1], num_groups=32, eps=1e-6, device=device, dtype=dtype),
                SiLU(),
                Conv2d(
                    in_channels=resnet_sizes[-1],
                    out_channels=latent_dim,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
            Chain(
                Conv2d(in_channels=8, out_channels=8, kernel_size=1, device=device, dtype=dtype),
                Slicing(dim=1, end=4),
            ),
        )

    def init_context(self) -> Contexts:
        return {"sampling": {"shapes": []}}


class Decoder(Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.resnet_sizes: list[int] = [128, 256, 512, 512, 512]
        self.latent_dim: int = 4
        self.output_channels: int = 3
        resnet_sizes = self.resnet_sizes[::-1]
        resnet_layers: list[Chain] = [
            (
                Chain(
                    [
                        Resnet(
                            in_channels=resnet_sizes[i - 1] if i > 0 else resnet_sizes[0],
                            out_channels=resnet_sizes[i],
                            device=device,
                            dtype=dtype,
                        ),
                        Resnet(in_channels=resnet_sizes[i], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                        Resnet(in_channels=resnet_sizes[i], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                    ]
                )
                if i > 0
                else Chain(
                    [
                        Resnet(in_channels=resnet_sizes[0], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                        Resnet(in_channels=resnet_sizes[i], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                    ]
                )
            )
            for i in range(len(resnet_sizes))
        ]
        attention_layer = Residual(
            GroupNorm(channels=resnet_sizes[0], num_groups=32, eps=1e-6, device=device, dtype=dtype),
            SelfAttention2d(channels=resnet_sizes[0], device=device, dtype=dtype),
        )
        resnet_layers[0].insert(1, attention_layer)
        for _, layer in zip(range(3), resnet_layers[1:]):
            channels: int = layer[-1].out_channels
            layer.insert(-1, Upsample(channels=channels, upsample_factor=2, device=device, dtype=dtype))
        super().__init__(
            Conv2d(
                in_channels=self.latent_dim, out_channels=self.latent_dim, kernel_size=1, device=device, dtype=dtype
            ),
            Conv2d(
                in_channels=self.latent_dim,
                out_channels=resnet_sizes[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            Chain(*resnet_layers),
            Chain(
                GroupNorm(channels=resnet_sizes[-1], num_groups=32, eps=1e-6, device=device, dtype=dtype),
                SiLU(),
                Conv2d(
                    in_channels=resnet_sizes[-1],
                    out_channels=self.output_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class LatentDiffusionAutoencoder(Chain):
    encoder_scale = 0.18125

    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            Encoder(device=device, dtype=dtype),
            Decoder(device=device, dtype=dtype),
        )

    def encode(self, x: Tensor) -> Tensor:
        encoder = self[0]
        x = self.encoder_scale * encoder(x)
        return x

    def decode(self, x: Tensor) -> Tensor:
        decoder = self[1]
        x = decoder(x / self.encoder_scale)
        return x

    def encode_image(self, image: Image.Image) -> Tensor:
        x = image_to_tensor(image, device=self.device, dtype=self.dtype)
        x = 2 * x - 1
        return self.encode(x)

    def decode_latents(self, x: Tensor) -> Image.Image:
        x = self.decode(x)
        x = (x + 1) / 2
        return tensor_to_image(x)

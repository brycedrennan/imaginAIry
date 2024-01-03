from typing import Iterable, cast

from torch import Tensor, device as Device, dtype as DType

from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter
from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.fluxion.layers import Chain, Conv2d, Lambda, Passthrough, Residual, SiLU, Slicing, UseContext
from imaginairy.vendored.refiners.foundationals.latent_diffusion.range_adapter import RangeAdapter2d
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import (
    DownBlocks,
    MiddleBlock,
    ResidualBlock,
    SD1UNet,
    TimestepEncoder,
)


class ConditionEncoder(Chain):
    """Encode an image to be used as a condition for Controlnet.

    Input is a `batch 3 width height` tensor, output is a `batch 320 width//8 height//8` tensor.
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.out_channels = (16, 32, 96, 256)
        super().__init__(
            Chain(
                Conv2d(
                    in_channels=3,
                    out_channels=self.out_channels[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                SiLU(),
            ),
            *(
                Chain(
                    Conv2d(
                        in_channels=self.out_channels[i],
                        out_channels=self.out_channels[i],
                        kernel_size=3,
                        padding=1,
                        device=device,
                        dtype=dtype,
                    ),
                    SiLU(),
                    Conv2d(
                        in_channels=self.out_channels[i],
                        out_channels=self.out_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        device=device,
                        dtype=dtype,
                    ),
                    SiLU(),
                )
                for i in range(len(self.out_channels) - 1)
            ),
            Conv2d(
                in_channels=self.out_channels[-1],
                out_channels=320,
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
        )


class Controlnet(Passthrough):
    def __init__(
        self, name: str, scale: float = 1.0, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        """Controlnet is a Half-UNet that collects residuals from the UNet and uses them to condition the UNet.

        Input is a `batch 3 width height` tensor, output is a `batch 1280 width//8 height//8` tensor with residuals
        stored in the context.

        It has to use the same context as the UNet: `unet` and `sampling`.
        """
        self.name = name
        self.scale = scale
        super().__init__(
            TimestepEncoder(context_key=f"timestep_embedding_{name}", device=device, dtype=dtype),
            Slicing(dim=1, end=4),  # support inpainting
            DownBlocks(in_channels=4, device=device, dtype=dtype),
            MiddleBlock(device=device, dtype=dtype),
        )

        # We run the condition encoder at each step. Caching the result
        # is not worth it as subsequent runs take virtually no time (FG-374).
        self.DownBlocks[0].append(
            Residual(
                UseContext("controlnet", f"condition_{name}"),
                ConditionEncoder(device=device, dtype=dtype),
            ),
        )
        for residual_block in self.layers(ResidualBlock):
            chain = residual_block.Chain
            RangeAdapter2d(
                target=chain.Conv2d_1,
                channels=residual_block.out_channels,
                embedding_dim=1280,
                context_key=f"timestep_embedding_{name}",
                device=device,
                dtype=dtype,
            ).inject(chain)
        for n, block in enumerate(cast(Iterable[Chain], self.DownBlocks)):
            assert hasattr(block[0], "out_channels"), (
                "The first block of every subchain in DownBlocks is expected to respond to `out_channels`,"
                f" {block[0]} does not."
            )
            out_channels: int = block[0].out_channels
            block.append(
                Passthrough(
                    Conv2d(
                        in_channels=out_channels, out_channels=out_channels, kernel_size=1, device=device, dtype=dtype
                    ),
                    Lambda(self._store_nth_residual(n)),
                )
            )
        self.MiddleBlock.append(
            Passthrough(
                Conv2d(in_channels=1280, out_channels=1280, kernel_size=1, device=device, dtype=dtype),
                Lambda(self._store_nth_residual(12)),
            )
        )

    def _store_nth_residual(self, n: int):
        def _store_residual(x: Tensor):
            residuals = self.use_context("unet")["residuals"]
            residuals[n] = residuals[n] + x * self.scale
            return x

        return _store_residual


class SD1ControlnetAdapter(Chain, Adapter[SD1UNet]):
    def __init__(
        self, target: SD1UNet, name: str, scale: float = 1.0, weights: dict[str, Tensor] | None = None
    ) -> None:
        self.name = name

        controlnet = Controlnet(name=name, scale=scale, device=target.device, dtype=target.dtype)
        if weights is not None:
            controlnet.load_state_dict(weights)
        self._controlnet: list[Controlnet] = [controlnet]  # not registered by PyTorch

        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self: "SD1ControlnetAdapter", parent: Chain | None = None) -> "SD1ControlnetAdapter":
        controlnet = self._controlnet[0]
        target_controlnets = [x for x in self.target if isinstance(x, Controlnet)]
        assert controlnet not in target_controlnets, f"{controlnet} is already injected"
        for cn in target_controlnets:
            assert cn.name != self.name, f"Controlnet named {self.name} is already injected"
        self.target.insert(0, controlnet)
        return super().inject(parent)

    def eject(self) -> None:
        self.target.remove(self._controlnet[0])
        super().eject()

    def init_context(self) -> Contexts:
        return {"controlnet": {f"condition_{self.name}": None}}

    def set_scale(self, scale: float) -> None:
        self._controlnet[0].scale = scale

    def set_controlnet_condition(self, condition: Tensor) -> None:
        self.set_context("controlnet", {f"condition_{self.name}": condition})

    def structural_copy(self: "SD1ControlnetAdapter") -> "SD1ControlnetAdapter":
        raise RuntimeError("Controlnet cannot be copied, eject it first.")

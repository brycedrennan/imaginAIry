from torch import Tensor

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import ResidualAccumulator
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet
from imaginairy.vendored.refiners.foundationals.latent_diffusion.t2i_adapter import ConditionEncoderXL, T2IAdapter, T2IFeatures


class SDXLT2IAdapter(T2IAdapter[SDXLUNet]):
    def __init__(
        self,
        target: SDXLUNet,
        name: str,
        condition_encoder: ConditionEncoderXL | None = None,
        scale: float = 1.0,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        self.residual_indices = (3, 5, 8)  # the UNet's middle block is handled separately (see `inject` and `eject`)
        self._features = [T2IFeatures(name=name, index=i, scale=scale) for i in range(4)]
        super().__init__(
            target=target,
            name=name,
            condition_encoder=condition_encoder or ConditionEncoderXL(device=target.device, dtype=target.dtype),
            weights=weights,
        )

    def inject(self: "SDXLT2IAdapter", parent: fl.Chain | None = None) -> "SDXLT2IAdapter":
        def sanity_check_t2i(block: fl.Chain) -> None:
            for t2i_layer in block.layers(layer_type=T2IFeatures):
                assert t2i_layer.name != self.name, f"T2I-Adapter named {self.name} is already injected"

        # Note: `strict=False` because `residual_indices` is shorter than `_features` due to MiddleBlock (see below)
        for n, feat in zip(self.residual_indices, self._features, strict=False):
            block = self.target.DownBlocks[n]
            sanity_check_t2i(block)
            block.insert_before_type(ResidualAccumulator, feat)

        # Special case: the MiddleBlock has no ResidualAccumulator (this is done via a subsequent layer) so just append
        sanity_check_t2i(self.target.MiddleBlock)
        self.target.MiddleBlock.append(self._features[-1])
        return super().inject(parent)

    def eject(self: "SDXLT2IAdapter") -> None:
        # See `inject` re: `strict=False`
        for n, feat in zip(self.residual_indices, self._features, strict=False):
            self.target.DownBlocks[n].remove(feat)
        self.target.MiddleBlock.remove(self._features[-1])
        super().eject()

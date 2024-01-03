from torch import Tensor

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import ResidualAccumulator, SD1UNet
from imaginairy.vendored.refiners.foundationals.latent_diffusion.t2i_adapter import ConditionEncoder, T2IAdapter, T2IFeatures


class SD1T2IAdapter(T2IAdapter[SD1UNet]):
    def __init__(
        self,
        target: SD1UNet,
        name: str,
        condition_encoder: ConditionEncoder | None = None,
        scale: float = 1.0,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        self.residual_indices = (2, 5, 8, 11)
        self._features = [T2IFeatures(name=name, index=i, scale=scale) for i in range(4)]
        super().__init__(
            target=target,
            name=name,
            condition_encoder=condition_encoder or ConditionEncoder(device=target.device, dtype=target.dtype),
            weights=weights,
        )

    def inject(self: "SD1T2IAdapter", parent: fl.Chain | None = None) -> "SD1T2IAdapter":
        for n, feat in zip(self.residual_indices, self._features, strict=True):
            block = self.target.DownBlocks[n]
            for t2i_layer in block.layers(layer_type=T2IFeatures):
                assert t2i_layer.name != self.name, f"T2I-Adapter named {self.name} is already injected"
            block.insert_before_type(ResidualAccumulator, feat)
        return super().inject(parent)

    def eject(self: "SD1T2IAdapter") -> None:
        for n, feat in zip(self.residual_indices, self._features, strict=True):
            self.target.DownBlocks[n].remove(feat)
        super().eject()

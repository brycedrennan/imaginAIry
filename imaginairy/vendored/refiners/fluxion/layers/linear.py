from jaxtyping import Float
from torch import Tensor, device as Device, dtype as DType
from torch.nn import Linear as _Linear

from imaginairy.vendored.refiners.fluxion.layers.activations import ReLU
from imaginairy.vendored.refiners.fluxion.layers.chain import Chain
from imaginairy.vendored.refiners.fluxion.layers.module import Module, WeightedModule


class Linear(_Linear, WeightedModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(  # type: ignore
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Float[Tensor, "batch in_features"]) -> Float[Tensor, "batch out_features"]:  # type: ignore
        return super().forward(x)


class MultiLinear(Chain):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inner_dim: int,
        num_layers: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        layers: list[Module] = []
        for i in range(num_layers - 1):
            layers.append(Linear(input_dim if i == 0 else inner_dim, inner_dim, device=device, dtype=dtype))
            layers.append(ReLU())
        layers.append(Linear(inner_dim, output_dim, device=device, dtype=dtype))

        super().__init__(layers)

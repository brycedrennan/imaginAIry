from torch import Tensor, sigmoid
from torch.nn.functional import (
    gelu,  # type: ignore
    silu,
)

from imaginairy.vendored.refiners.fluxion.layers.module import Module


class Activation(Module):
    def __init__(self) -> None:
        super().__init__()


class SiLU(Activation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return silu(x)  # type: ignore


class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GeLU(Activation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)  # type: ignore


class ApproximateGeLU(Activation):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)
    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * sigmoid(1.702 * x)


class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class GLU(Activation):
    """
    Gated Linear Unit activation layer.

    See https://arxiv.org/abs/2002.05202v1 for details.
    """

    def __init__(self, activation: Activation) -> None:
        super().__init__()
        self.activation = activation

    def __repr__(self):
        return f"{self.__class__.__name__}(activation={self.activation})"

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0, "Non-batch input dimension must be divisible by 2"
        output, gate = x.chunk(2, dim=-1)
        return output * self.activation(gate)

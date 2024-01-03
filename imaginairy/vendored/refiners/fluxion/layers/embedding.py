from jaxtyping import Float, Int
from torch import Tensor, device as Device, dtype as DType
from torch.nn import Embedding as _Embedding

from imaginairy.vendored.refiners.fluxion.layers.module import WeightedModule


class Embedding(_Embedding, WeightedModule):  # type: ignore
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        _Embedding.__init__(  # type: ignore
            self, num_embeddings=num_embeddings, embedding_dim=embedding_dim, device=device, dtype=dtype
        )

    def forward(self, x: Int[Tensor, "batch length"]) -> Float[Tensor, "batch length embedding_dim"]:  # type: ignore
        return super().forward(x)

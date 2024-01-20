import math

from jaxtyping import Float, Int
from torch import Tensor, arange, cat, cos, device as Device, dtype as DType, exp, float32, sin

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter


def compute_sinusoidal_embedding(
    x: Int[Tensor, "*batch 1"],
    embedding_dim: int,
) -> Float[Tensor, "*batch 1 embedding_dim"]:
    half_dim = embedding_dim // 2
    # Note: it is important that this computation is done in float32.
    # The result can be cast to lower precision later if necessary.
    exponent = -math.log(10000) * arange(start=0, end=half_dim, dtype=float32, device=x.device)
    exponent /= half_dim
    embedding = x.unsqueeze(1).float() * exp(exponent).unsqueeze(0)
    embedding = cat([cos(embedding), sin(embedding)], dim=-1)
    return embedding


class RangeEncoder(fl.Chain):
    def __init__(
        self,
        sinusoidal_embedding_dim: int,
        embedding_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.sinusoidal_embedding_dim = sinusoidal_embedding_dim
        self.embedding_dim = embedding_dim
        super().__init__(
            fl.Lambda(self.compute_sinusoidal_embedding),
            fl.Converter(set_device=False, set_dtype=True),
            fl.Linear(in_features=sinusoidal_embedding_dim, out_features=embedding_dim, device=device, dtype=dtype),
            fl.SiLU(),
            fl.Linear(in_features=embedding_dim, out_features=embedding_dim, device=device, dtype=dtype),
        )

    def compute_sinusoidal_embedding(self, x: Int[Tensor, "*batch 1"]) -> Float[Tensor, "*batch 1 embedding_dim"]:
        return compute_sinusoidal_embedding(x, embedding_dim=self.sinusoidal_embedding_dim)


class RangeAdapter2d(fl.Sum, Adapter[fl.Conv2d]):
    def __init__(
        self,
        target: fl.Conv2d,
        channels: int,
        embedding_dim: int,
        context_key: str,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.context_key = context_key
        with self.setup_adapter(target):
            super().__init__(
                target,
                fl.Chain(
                    fl.UseContext("range_adapter", context_key),
                    fl.SiLU(),
                    fl.Linear(in_features=embedding_dim, out_features=channels, device=device, dtype=dtype),
                    fl.View(-1, channels, 1, 1),
                ),
            )

from torch import Tensor, arange, device as Device, dtype as DType

import imaginairy.vendored.refiners.fluxion.layers as fl


class PositionalEncoder(fl.Chain):
    def __init__(
        self,
        max_sequence_length: int,
        embedding_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        super().__init__(
            fl.Lambda(func=self.get_position_ids),
            fl.Embedding(
                num_embeddings=max_sequence_length,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            ),
        )

    @property
    def position_ids(self) -> Tensor:
        return arange(end=self.max_sequence_length, device=self.device).reshape(1, -1)

    def get_position_ids(self, x: Tensor) -> Tensor:
        return self.position_ids[:, : x.shape[1]]


class FeedForward(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        super().__init__(
            fl.Linear(in_features=embedding_dim, out_features=feedforward_dim, device=device, dtype=dtype),
            fl.GeLU(),
            fl.Linear(in_features=feedforward_dim, out_features=embedding_dim, device=device, dtype=dtype),
        )

from collections.abc import Sequence
from enum import Enum, auto

import torch
from jaxtyping import Float, Int
from torch import Tensor, device as Device, dtype as DType, nn

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.context import Contexts


class CoordinateEncoder(fl.Chain):
    def __init__(
        self,
        num_positional_features: int = 64,
        scale: float = 1,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.num_positional_features = num_positional_features
        self.scale = scale

        super().__init__(
            fl.Multiply(scale=2, bias=-1),
            fl.Linear(in_features=2, out_features=num_positional_features, bias=False, device=device, dtype=dtype),
            fl.Multiply(scale=2 * torch.pi * self.scale),
            fl.Concatenate(fl.Sin(), fl.Cos(), dim=-1),
        )


class PointType(Enum):
    BACKGROUND = auto()
    FOREGROUND = auto()
    BOX_TOP_LEFT = auto()
    BOX_BOTTOM_RIGHT = auto()
    NOT_A_POINT = auto()


class PointTypeEmbedding(fl.WeightedModule, fl.ContextModule):
    def __init__(self, embedding_dim: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(data=torch.randn(len(PointType), self.embedding_dim, device=device, dtype=dtype))

    def forward(self, type_mask: Int[Tensor, "1 num_points"]) -> Float[Tensor, "1 num_points embedding_dim"]:
        assert isinstance(type_mask, Tensor), "type_mask must be a Tensor."

        embeddings = torch.zeros(*type_mask.shape, self.embedding_dim).to(device=type_mask.device)
        for type_id in PointType:
            mask = type_mask == type_id.value
            embeddings[mask] = self.weight[type_id.value - 1]

        return embeddings


class PointEncoder(fl.Chain):
    def __init__(
        self, embedding_dim: int = 256, scale: float = 1, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        assert embedding_dim % 2 == 0, "embedding_dim must be divisible by 2."
        self.embedding_dim = embedding_dim
        self.scale = scale

        super().__init__(
            CoordinateEncoder(num_positional_features=embedding_dim // 2, scale=scale, device=device, dtype=dtype),
            fl.Lambda(func=self.pad),
            fl.Residual(
                fl.UseContext(context="point_encoder", key="type_mask"),
                PointTypeEmbedding(embedding_dim=embedding_dim, device=device, dtype=dtype),
            ),
        )

    def pad(self, x: Tensor) -> Tensor:
        type_mask: Tensor = self.use_context("point_encoder")["type_mask"]
        if torch.any((type_mask == PointType.BOX_TOP_LEFT.value) | (type_mask == PointType.BOX_BOTTOM_RIGHT.value)):
            # Some boxes have been passed: no need to pad in this case
            return x
        type_mask = torch.cat(
            [type_mask, torch.full((type_mask.shape[0], 1), PointType.NOT_A_POINT.value, device=type_mask.device)],
            dim=1,
        )
        self.set_context(context="point_encoder", value={"type_mask": type_mask})
        return torch.cat([x, torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device)], dim=1)

    def init_context(self) -> Contexts:
        return {
            "point_encoder": {
                "type_mask": None,
            }
        }

    def set_type_mask(self, type_mask: Int[Tensor, "1 num_points"]) -> None:
        self.set_context(context="point_encoder", value={"type_mask": type_mask})

    def get_dense_positional_embedding(
        self, image_embedding_size: tuple[int, int]
    ) -> Float[Tensor, "num_positional_features height width"]:
        coordinate_encoder = self.ensure_find(layer_type=CoordinateEncoder)
        height, width = image_embedding_size
        grid = torch.ones((height, width), device=self.device, dtype=torch.float32)
        y_embedding = grid.cumsum(dim=0) - 0.5
        x_embedding = grid.cumsum(dim=1) - 0.5
        y_embedding = y_embedding / height
        x_embedding = x_embedding / width
        positional_embedding = (
            coordinate_encoder(torch.stack(tensors=[x_embedding, y_embedding], dim=-1))
            .permute(2, 0, 1)
            .unsqueeze(dim=0)
        )
        return positional_embedding

    def points_to_tensor(
        self,
        foreground_points: Sequence[tuple[float, float]] | None = None,
        background_points: Sequence[tuple[float, float]] | None = None,
        not_a_points: Sequence[tuple[float, float]] | None = None,
        box_points: Sequence[Sequence[tuple[float, float]]] | None = None,
    ) -> tuple[Float[Tensor, "1 num_points 2"], Int[Tensor, "1 num_points"]]:
        foreground_points = foreground_points or []
        background_points = background_points or []
        not_a_points = not_a_points or []
        box_points = box_points or []
        top_left_points = [box[0] for box in box_points]
        bottom_right_points = [box[1] for box in box_points]
        coordinates: list[Tensor] = []
        type_ids: list[Tensor] = []

        # Must be in sync with PointType enum
        for type_id, coords_seq in zip(
            PointType, [background_points, foreground_points, top_left_points, bottom_right_points, not_a_points]
        ):
            if len(coords_seq) > 0:
                coords_tensor = torch.tensor(data=list(coords_seq), dtype=torch.float, device=self.device)
                coordinates.append(coords_tensor)
                point_ids = torch.tensor(data=[type_id.value] * len(coords_seq), dtype=torch.int, device=self.device)
                type_ids.append(point_ids)

        all_coordinates = torch.cat(tensors=coordinates, dim=0).unsqueeze(dim=0)
        type_mask = torch.cat(tensors=type_ids, dim=0).unsqueeze(dim=0)

        return all_coordinates, type_mask


class MaskEncoder(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 256,
        intermediate_channels: int = 16,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.intermediate_channels = intermediate_channels
        super().__init__(
            fl.Conv2d(
                in_channels=1,
                out_channels=self.intermediate_channels // 4,
                kernel_size=2,
                stride=2,
                device=device,
                dtype=dtype,
            ),
            fl.LayerNorm2d(channels=self.intermediate_channels // 4, device=device, dtype=dtype),
            fl.GeLU(),
            fl.Conv2d(
                in_channels=self.intermediate_channels // 4,
                out_channels=self.intermediate_channels,
                kernel_size=2,
                stride=2,
                device=device,
                dtype=dtype,
            ),
            fl.LayerNorm2d(channels=self.intermediate_channels, device=device, dtype=dtype),
            fl.GeLU(),
            fl.Conv2d(
                in_channels=self.intermediate_channels,
                out_channels=self.embedding_dim,
                kernel_size=1,
                device=device,
                dtype=dtype,
            ),
        )
        self.register_parameter(
            "no_mask_embedding", nn.Parameter(torch.randn(1, embedding_dim, device=device, dtype=dtype))
        )

    def get_no_mask_dense_embedding(
        self, image_embedding_size: tuple[int, int], batch_size: int = 1
    ) -> Float[Tensor, "batch embedding_dim image_embedding_height image_embedding_width"]:
        return self.no_mask_embedding.reshape(1, -1, 1, 1).expand(
            batch_size, -1, image_embedding_size[0], image_embedding_size[1]
        )

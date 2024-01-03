import torch
from torch import Tensor, device as Device, dtype as DType, nn

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.foundationals.segment_anything.transformer import (
    SparseCrossDenseAttention,
    TwoWayTranformerLayer,
)


class EmbeddingsAggregator(fl.ContextModule):
    def __init__(self, num_output_mask: int = 3) -> None:
        super().__init__()
        self.num_mask_tokens = num_output_mask

    def forward(self, iou_mask_tokens: Tensor) -> Tensor:
        mask_decoder = self.ensure_parent
        mask_decoder_context = mask_decoder.use_context(context_name="mask_decoder")
        image_embedding = mask_decoder_context["image_embedding"]
        point_embedding = mask_decoder_context["point_embedding"]
        mask_embedding = mask_decoder_context["mask_embedding"]
        dense_positional_embedding = mask_decoder_context["dense_positional_embedding"]

        sparse_embedding = torch.cat(tensors=(iou_mask_tokens, point_embedding), dim=1)
        dense_embedding = (image_embedding + mask_embedding).flatten(start_dim=2).transpose(1, 2)
        if dense_positional_embedding.shape != dense_embedding.shape:
            dense_positional_embedding = dense_positional_embedding.flatten(start_dim=2).transpose(1, 2)

        mask_decoder_context.update(
            {
                "dense_embedding": dense_embedding,
                "dense_positional_embedding": dense_positional_embedding,
                "sparse_embedding": sparse_embedding,
            }
        )
        mask_decoder.set_context(context="mask_decoder", value=mask_decoder_context)

        return sparse_embedding


class Transformer(fl.Chain):
    pass


class Hypernetworks(fl.Concatenate):
    def __init__(
        self,
        embedding_dim: int = 256,
        num_layers: int = 3,
        num_mask_tokens: int = 3,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_mask_tokens = num_mask_tokens

        super().__init__(
            *[
                fl.Chain(
                    fl.Slicing(dim=1, start=i + 1, end=i + 2),
                    fl.MultiLinear(
                        input_dim=embedding_dim,
                        output_dim=embedding_dim // 8,
                        inner_dim=embedding_dim,
                        num_layers=num_layers,
                        device=device,
                        dtype=dtype,
                    ),
                )
                for i in range(num_mask_tokens + 1)
            ],
            dim=1,
        )


class DenseEmbeddingUpscaling(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 256,
        dense_embedding_side_dim: int = 64,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dense_embedding_side_dim = dense_embedding_side_dim

        super().__init__(
            fl.UseContext(context="mask_decoder", key="dense_embedding"),
            fl.Transpose(dim0=1, dim1=2),
            fl.Reshape(embedding_dim, dense_embedding_side_dim, dense_embedding_side_dim),
            fl.ConvTranspose2d(
                in_channels=embedding_dim,
                out_channels=embedding_dim // 4,
                kernel_size=2,
                stride=2,
                device=device,
                dtype=dtype,
            ),
            fl.LayerNorm2d(channels=embedding_dim // 4, device=device, dtype=dtype),
            fl.GeLU(),
            fl.ConvTranspose2d(
                in_channels=embedding_dim // 4,
                out_channels=embedding_dim // 8,
                kernel_size=2,
                stride=2,
                device=device,
                dtype=dtype,
            ),
            fl.GeLU(),
            fl.Flatten(start_dim=2),
        )


class IOUMaskEncoder(fl.WeightedModule):
    def __init__(
        self,
        embedding_dim: int = 256,
        num_mask_tokens: int = 4,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_mask_tokens = num_mask_tokens
        # aka prompt tokens + output token (for IoU scores prediction)
        self.weight = nn.Parameter(data=torch.randn(num_mask_tokens + 1, embedding_dim, device=device, dtype=dtype))

    def forward(self) -> Tensor:
        return self.weight.unsqueeze(dim=0)


class MaskPrediction(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        num_mask_tokens: int,
        num_layers: int = 3,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_mask_tokens = num_mask_tokens
        self.num_layers = num_layers
        super().__init__(
            fl.Matmul(
                input=Hypernetworks(
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    num_mask_tokens=num_mask_tokens,
                    device=device,
                    dtype=dtype,
                ),
                other=DenseEmbeddingUpscaling(embedding_dim=embedding_dim, device=device, dtype=dtype),
            ),
            fl.Slicing(dim=1, start=1),
            fl.Reshape(num_mask_tokens, embedding_dim, embedding_dim),
        )


class IOUPrediction(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        num_mask_tokens: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        super().__init__(
            fl.Slicing(dim=1, start=0, end=1),
            fl.Squeeze(dim=0),
            fl.MultiLinear(
                input_dim=embedding_dim,
                output_dim=num_mask_tokens + 1,
                inner_dim=embedding_dim,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            ),
            fl.Slicing(dim=-1, start=1),
        )


class MaskDecoder(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 256,
        feed_forward_dim: int = 2048,
        num_layers: int = 2,
        num_output_mask: int = 3,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_mask_tokens = num_output_mask
        self.feed_forward_dim = feed_forward_dim
        self.num_layers = num_layers

        super().__init__(
            IOUMaskEncoder(
                embedding_dim=embedding_dim, num_mask_tokens=num_output_mask + 1, device=device, dtype=dtype
            ),
            EmbeddingsAggregator(num_output_mask=num_output_mask),
            Transformer(
                *(
                    TwoWayTranformerLayer(
                        embedding_dim=embedding_dim,
                        num_heads=8,
                        feed_forward_dim=feed_forward_dim,
                        use_residual_self_attention=i > 0,
                        device=device,
                        dtype=dtype,
                    )
                    for i in range(num_layers)
                ),
                SparseCrossDenseAttention(embedding_dim=embedding_dim, device=device, dtype=dtype),
                fl.LayerNorm(normalized_shape=embedding_dim, device=device, dtype=dtype),
            ),
            fl.Parallel(
                MaskPrediction(
                    embedding_dim=embedding_dim, num_mask_tokens=num_output_mask, device=device, dtype=dtype
                ),
                IOUPrediction(
                    embedding_dim=embedding_dim,
                    num_layers=3,
                    num_mask_tokens=num_output_mask,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )

    def init_context(self) -> Contexts:
        return {
            "mask_decoder": {
                "image_embedding": None,
                "point_embedding": None,
                "mask_embedding": None,
                "dense_positional_embedding": None,
            }
        }

    def set_image_embedding(self, image_embedding: Tensor) -> None:
        mask_decoder_context = self.use_context(context_name="mask_decoder")
        mask_decoder_context["image_embedding"] = image_embedding

    def set_point_embedding(self, point_embedding: Tensor) -> None:
        mask_decoder_context = self.use_context(context_name="mask_decoder")
        mask_decoder_context["point_embedding"] = point_embedding

    def set_mask_embedding(self, mask_embedding: Tensor) -> None:
        mask_decoder_context = self.use_context(context_name="mask_decoder")
        mask_decoder_context["mask_embedding"] = mask_embedding

    def set_dense_positional_embedding(self, dense_positional_embedding: Tensor) -> None:
        mask_decoder_context = self.use_context(context_name="mask_decoder")
        mask_decoder_context["dense_positional_embedding"] = dense_positional_embedding

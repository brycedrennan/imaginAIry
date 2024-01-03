import torch
from torch import Tensor

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.layers.activations import Activation


class ClassToken(fl.Chain):
    """Learnable token representing the class of the input."""

    def __init__(
        self,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim

        super().__init__(
            fl.Parameter(
                *(1, embedding_dim),
                device=device,
                dtype=dtype,
            ),
        )


class PositionalEncoder(fl.Residual):
    """Encode the position of each patch in the input."""

    def __init__(
        self,
        sequence_length: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.num_patches = sequence_length
        self.embedding_dim = embedding_dim

        super().__init__(
            fl.Parameter(
                *(sequence_length, embedding_dim),
                device=device,
                dtype=dtype,
            ),
        )


class LayerScale(fl.WeightedModule):
    """Scale the input tensor by a learnable parameter."""

    def __init__(
        self,
        embedding_dim: int,
        init_value: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.register_parameter(
            name="weight",
            param=torch.nn.Parameter(
                torch.full(
                    size=(embedding_dim,),
                    fill_value=init_value,
                    dtype=dtype,
                    device=device,
                ),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


class FeedForward(fl.Chain):
    """Apply two linear transformations interleaved by an activation function."""

    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        activation: Activation = fl.GeLU,  # type: ignore
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim

        super().__init__(
            fl.Linear(
                in_features=embedding_dim,
                out_features=feedforward_dim,
                device=device,
                dtype=dtype,
            ),
            activation(),
            fl.Linear(
                in_features=feedforward_dim,
                out_features=embedding_dim,
                device=device,
                dtype=dtype,
            ),
        )


class PatchEncoder(fl.Chain):
    """Encode an image into a sequence of patches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        super().__init__(
            fl.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                device=device,
                dtype=dtype,
            ),  # (N,3,H,W) -> (N,D,P,P)
            fl.Reshape(out_channels, -1),  # (N,D,P,P) -> (N,D,P²)
            fl.Transpose(1, 2),  # (N,D,P²) -> (N,P²,D)
        )


class TransformerLayer(fl.Chain):
    """Apply a multi-head self-attention mechanism to the input tensor."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        norm_eps: float,
        mlp_ratio: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.mlp_ratio = mlp_ratio

        super().__init__(
            fl.Residual(
                fl.LayerNorm(
                    normalized_shape=embedding_dim,
                    eps=norm_eps,
                    device=device,
                    dtype=dtype,
                ),
                fl.SelfAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    device=device,
                    dtype=dtype,
                ),
                LayerScale(
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Residual(
                fl.LayerNorm(
                    normalized_shape=embedding_dim,
                    eps=norm_eps,
                    device=device,
                    dtype=dtype,
                ),
                FeedForward(
                    embedding_dim=embedding_dim,
                    feedforward_dim=embedding_dim * mlp_ratio,
                    device=device,
                    dtype=dtype,
                ),
                LayerScale(
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class Transformer(fl.Chain):
    """Alias for a Chain of TransformerLayer."""


class Registers(fl.Concatenate):
    """Insert register tokens between CLS token and patches."""

    def __init__(
        self,
        num_registers: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.num_registers = num_registers
        self.embedding_dim = embedding_dim

        super().__init__(
            fl.Slicing(dim=1, end=1),
            fl.Parameter(
                *(num_registers, embedding_dim),
                device=device,
                dtype=dtype,
            ),
            fl.Slicing(dim=1, start=1),
            dim=1,
        )


class ViT(fl.Chain):
    """Vision Transformer (ViT).

    see https://arxiv.org/abs/2010.11929v2
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        patch_size: int = 16,
        image_size: int = 224,
        num_layers: int = 12,
        num_heads: int = 12,
        norm_eps: float = 1e-6,
        mlp_ratio: int = 4,
        num_registers: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        num_patches = image_size // patch_size
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.mlp_ratio = mlp_ratio
        self.num_registers = num_registers

        super().__init__(
            fl.Concatenate(
                ClassToken(
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                PatchEncoder(
                    in_channels=3,
                    out_channels=embedding_dim,
                    patch_size=patch_size,
                    device=device,
                    dtype=dtype,
                ),
                dim=1,
            ),
            # TODO: support https://github.com/facebookresearch/dinov2/blob/2302b6b/dinov2/models/vision_transformer.py#L179
            PositionalEncoder(
                sequence_length=num_patches**2 + 1,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            ),
            Transformer(
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
            fl.LayerNorm(
                normalized_shape=embedding_dim,
                eps=norm_eps,
                device=device,
                dtype=dtype,
            ),
        )

        if self.num_registers > 0:
            registers = Registers(
                num_registers=num_registers,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )
            self.insert_before_type(Transformer, registers)


class ViT_tiny(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=192,
            patch_size=16,
            image_size=224,
            num_layers=12,
            num_heads=3,
            device=device,
            dtype=dtype,
        )


class ViT_small(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=384,
            patch_size=16,
            image_size=224,
            num_layers=12,
            num_heads=6,
            device=device,
            dtype=dtype,
        )


class ViT_base(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=768,
            patch_size=16,
            image_size=224,
            num_layers=12,
            num_heads=12,
            device=device,
            dtype=dtype,
        )


class ViT_large(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=1024,
            patch_size=16,
            image_size=224,
            num_layers=24,
            num_heads=16,
            device=device,
            dtype=dtype,
        )

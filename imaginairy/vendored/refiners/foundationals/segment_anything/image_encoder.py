import torch
from torch import Tensor, device as Device, dtype as DType, nn

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.fluxion.utils import pad


class PatchEncoder(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        use_bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_bias = use_bias
        super().__init__(
            fl.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size),
                use_bias=self.use_bias,
                device=device,
                dtype=dtype,
            ),
            fl.Permute(0, 2, 3, 1),
        )


class PositionalEncoder(fl.Residual):
    def __init__(
        self,
        embedding_dim: int,
        image_embedding_size: tuple[int, int],
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.image_embedding_size = image_embedding_size
        super().__init__(
            fl.Parameter(
                image_embedding_size[0],
                image_embedding_size[1],
                embedding_dim,
                device=device,
                dtype=dtype,
            ),
        )


class RelativePositionAttention(fl.WeightedModule):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        spatial_size: tuple[int, int],
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.spatial_size = spatial_size
        self.horizontal_embedding = nn.Parameter(
            data=torch.zeros(2 * spatial_size[0] - 1, self.head_dim, device=device, dtype=dtype)
        )
        self.vertical_embedding = nn.Parameter(
            data=torch.zeros(2 * spatial_size[1] - 1, self.head_dim, device=device, dtype=dtype)
        )

    @property
    def device(self) -> Device:
        return self.horizontal_embedding.device

    @property
    def dtype(self) -> DType:
        return self.horizontal_embedding.dtype

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, _ = x.shape
        x = (
            x.reshape(batch, width * height, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
            .reshape(3, batch * self.num_heads, width * height, -1)
        )
        query, key, value = x.unbind(dim=0)
        horizontal_relative_embedding, vertical_relative_embedding = self.compute_relative_embedding(x=query)
        attention = (query * self.head_dim**-0.5) @ key.transpose(dim0=-2, dim1=-1)
        # Order of operations is important here
        attention = (
            (attention.reshape(-1, height, width, height, width) + vertical_relative_embedding)
            + horizontal_relative_embedding
        ).reshape(attention.shape)
        attention = attention.softmax(dim=-1)
        attention = attention @ value
        attention = (
            attention.reshape(batch, self.num_heads, height, width, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch, height, width, -1)
        )
        return attention

    def compute_relative_coords(self, size: int) -> Tensor:
        x, y = torch.meshgrid(torch.arange(end=size), torch.arange(end=size), indexing="ij")
        return x - y + size - 1

    def compute_relative_embedding(self, x: Tensor) -> tuple[Tensor, Tensor]:
        width, height = self.spatial_size
        horizontal_coords = self.compute_relative_coords(size=width)
        vertical_coords = self.compute_relative_coords(size=height)
        horizontal_positional_embedding = self.horizontal_embedding[horizontal_coords]
        vertical_positional_embedding = self.vertical_embedding[vertical_coords]
        x = x.reshape(x.shape[0], width, height, -1)
        horizontal_relative_embedding = torch.einsum("bhwc,wkc->bhwk", x, horizontal_positional_embedding).unsqueeze(
            dim=-2
        )
        vertical_relative_embedding = torch.einsum("bhwc,hkc->bhwk", x, vertical_positional_embedding).unsqueeze(dim=-1)

        return horizontal_relative_embedding, vertical_relative_embedding


class FusedSelfAttention(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 768,
        spatial_size: tuple[int, int] = (64, 64),
        num_heads: int = 1,
        use_bias: bool = True,
        is_causal: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        assert (
            embedding_dim % num_heads == 0
        ), f"Embedding dim (embedding_dim={embedding_dim}) must be divisible by num heads (num_heads={num_heads})"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.is_causal = is_causal
        super().__init__(
            fl.Linear(
                in_features=self.embedding_dim,
                out_features=3 * self.embedding_dim,
                bias=self.use_bias,
                device=device,
                dtype=dtype,
            ),
            RelativePositionAttention(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                spatial_size=spatial_size,
                device=device,
                dtype=dtype,
            ),
            fl.Linear(
                in_features=self.embedding_dim,
                out_features=self.embedding_dim,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )


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
            fl.Linear(
                in_features=self.embedding_dim,
                out_features=self.feedforward_dim,
                bias=True,
                device=device,
                dtype=dtype,
            ),
            fl.GeLU(),
            fl.Linear(
                in_features=self.feedforward_dim,
                out_features=self.embedding_dim,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )


class WindowPartition(fl.ContextModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        context = self.use_context(context_name="window_partition")
        context.update({"original_height": height, "original_width": width})
        window_size = context["window_size"]
        padding_height = (window_size - height % window_size) % window_size
        padding_width = (window_size - width % window_size) % window_size
        if padding_height > 0 or padding_width > 0:
            x = pad(x=x, pad=(0, 0, 0, padding_width, 0, padding_height))
        padded_height, padded_width = height + padding_height, width + padding_width
        context.update({"padded_height": padded_height, "padded_width": padded_width})
        x = x.view(batch, padded_height // window_size, window_size, padded_width // window_size, window_size, channels)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
        return windows


class WindowMerge(fl.ContextModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        context = self.use_context(context_name="window_partition")
        window_size = context["window_size"]
        padded_height, padded_width = context["padded_height"], context["padded_width"]
        original_height, original_width = context["original_height"], context["original_width"]
        batch_size = x.shape[0] // (padded_height * padded_width // window_size // window_size)
        x = x.view(batch_size, padded_height // window_size, padded_width // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, padded_height, padded_width, -1)
        if padded_height > original_height or padded_width > original_width:
            x = x[:, :original_height, :original_width, :].contiguous()
        return x


class TransformerLayer(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        image_embedding_size: tuple[int, int],
        window_size: int | None = None,
        layer_norm_eps: float = 1e-6,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.window_size = window_size
        self.layer_norm_eps = layer_norm_eps
        self.image_embedding_size = image_embedding_size
        attention_spatial_size = (window_size, window_size) if window_size is not None else image_embedding_size
        reshape_or_merge = (
            WindowMerge()
            if self.window_size is not None
            else fl.Reshape(self.image_embedding_size[0], self.image_embedding_size[1], embedding_dim)
        )
        super().__init__(
            fl.Residual(
                fl.LayerNorm(normalized_shape=embedding_dim, eps=self.layer_norm_eps, device=device, dtype=dtype),
                WindowPartition() if self.window_size is not None else fl.Identity(),
                FusedSelfAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    spatial_size=attention_spatial_size,
                    device=device,
                    dtype=dtype,
                ),
                reshape_or_merge,
            ),
            fl.Residual(
                fl.LayerNorm(normalized_shape=embedding_dim, eps=self.layer_norm_eps, device=device, dtype=dtype),
                FeedForward(embedding_dim=embedding_dim, feedforward_dim=feedforward_dim, device=device, dtype=dtype),
            ),
        )

    def init_context(self) -> Contexts:
        return {"window_partition": {"window_size": self.window_size}}


class Neck(fl.Chain):
    def __init__(self, in_channels: int = 768, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.in_channels = in_channels
        super().__init__(
            fl.Permute(0, 3, 1, 2),
            fl.Conv2d(
                in_channels=self.in_channels,
                out_channels=256,
                kernel_size=1,
                use_bias=False,
                device=device,
                dtype=dtype,
            ),
            fl.LayerNorm2d(channels=256, device=device, dtype=dtype),
            fl.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                use_bias=False,
                device=device,
                dtype=dtype,
            ),
            fl.LayerNorm2d(channels=256, device=device, dtype=dtype),
        )


class Transformer(fl.Chain):
    pass


class SAMViT(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        global_attention_indices: tuple[int, ...] | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.image_size = (1024, 1024)
        self.patch_size = 16
        self.window_size = 14
        self.image_embedding_size = (self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)
        self.feed_forward_dim = 4 * self.embedding_dim
        self.global_attention_indices = global_attention_indices or tuple()
        super().__init__(
            PatchEncoder(
                in_channels=3, out_channels=embedding_dim, patch_size=self.patch_size, device=device, dtype=dtype
            ),
            PositionalEncoder(
                embedding_dim=embedding_dim, image_embedding_size=self.image_embedding_size, device=device, dtype=dtype
            ),
            Transformer(
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feedforward_dim=self.feed_forward_dim,
                    window_size=self.window_size if i not in self.global_attention_indices else None,
                    image_embedding_size=self.image_embedding_size,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ),
            Neck(in_channels=embedding_dim, device=device, dtype=dtype),
        )


class SAMViTH(SAMViT):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            embedding_dim=1280,
            num_layers=32,
            num_heads=16,
            global_attention_indices=(7, 15, 23, 31),
            device=device,
            dtype=dtype,
        )

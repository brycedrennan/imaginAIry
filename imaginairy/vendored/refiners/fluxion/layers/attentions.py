import math

import torch
from jaxtyping import Float
from torch import Tensor, device as Device, dtype as DType
from torch.nn.functional import scaled_dot_product_attention as _scaled_dot_product_attention  # type: ignore

from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.fluxion.layers.basics import Identity
from imaginairy.vendored.refiners.fluxion.layers.chain import Chain, Distribute, Lambda, Parallel
from imaginairy.vendored.refiners.fluxion.layers.linear import Linear
from imaginairy.vendored.refiners.fluxion.layers.module import Module


def scaled_dot_product_attention(
    query: Float[Tensor, "batch source_sequence_length dim"],
    key: Float[Tensor, "batch target_sequence_length dim"],
    value: Float[Tensor, "batch target_sequence_length dim"],
    is_causal: bool = False,
) -> Float[Tensor, "batch source_sequence_length dim"]:
    return _scaled_dot_product_attention(query, key, value, is_causal=is_causal)  # type: ignore


def sparse_dot_product_attention_non_optimized(
    query: Float[Tensor, "batch source_sequence_length dim"],
    key: Float[Tensor, "batch target_sequence_length dim"],
    value: Float[Tensor, "batch target_sequence_length dim"],
    is_causal: bool = False,
) -> Float[Tensor, "batch source_sequence_length dim"]:
    if is_causal:
        # TODO: implement causal attention
        raise NotImplementedError("Causal attention for non_optimized attention is not yet implemented")
    _, _, _, dim = query.shape
    attention = query @ key.permute(0, 1, 3, 2)
    attention = attention / math.sqrt(dim)
    attention = torch.softmax(input=attention, dim=-1)
    return attention @ value


class ScaledDotProductAttention(Module):
    def __init__(
        self,
        num_heads: int = 1,
        is_causal: bool | None = None,
        is_optimized: bool = True,
        slice_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.is_optimized = is_optimized
        self.slice_size = slice_size
        self.dot_product = (
            scaled_dot_product_attention if self.is_optimized else sparse_dot_product_attention_non_optimized
        )

    def forward(
        self,
        query: Float[Tensor, "batch num_queries embedding_dim"],
        key: Float[Tensor, "batch num_keys embedding_dim"],
        value: Float[Tensor, "batch num_values embedding_dim"],
        is_causal: bool | None = None,
    ) -> Float[Tensor, "batch num_queries dim"]:
        if self.slice_size is None:
            return self._process_attention(query, key, value, is_causal)

        return self._sliced_attention(query, key, value, is_causal=is_causal, slice_size=self.slice_size)

    def _sliced_attention(
        self,
        query: Float[Tensor, "batch num_queries embedding_dim"],
        key: Float[Tensor, "batch num_keys embedding_dim"],
        value: Float[Tensor, "batch num_values embedding_dim"],
        slice_size: int,
        is_causal: bool | None = None,
    ) -> Float[Tensor, "batch num_queries dim"]:
        _, num_queries, _ = query.shape
        output = torch.zeros_like(query)
        for start_idx in range(0, num_queries, slice_size):
            end_idx = min(start_idx + slice_size, num_queries)
            output[:, start_idx:end_idx, :] = self._process_attention(
                query[:, start_idx:end_idx, :], key, value, is_causal
            )
        return output

    def _process_attention(
        self,
        query: Float[Tensor, "batch num_queries embedding_dim"],
        key: Float[Tensor, "batch num_keys embedding_dim"],
        value: Float[Tensor, "batch num_values embedding_dim"],
        is_causal: bool | None = None,
    ) -> Float[Tensor, "batch num_queries dim"]:
        return self.merge_multi_head(
            x=self.dot_product(
                query=self.split_to_multi_head(query),
                key=self.split_to_multi_head(key),
                value=self.split_to_multi_head(value),
                is_causal=(
                    is_causal if is_causal is not None else (self.is_causal if self.is_causal is not None else False)
                ),
            )
        )

    def split_to_multi_head(
        self, x: Float[Tensor, "batch_size sequence_length embedding_dim"]
    ) -> Float[Tensor, "batch_size num_heads sequence_length (embedding_dim//num_heads)"]:
        assert (
            len(x.shape) == 3
        ), f"Expected tensor with shape (batch_size sequence_length embedding_dim), got {x.shape}"
        assert (
            x.shape[-1] % self.num_heads == 0
        ), f"Embedding dim (x.shape[-1]={x.shape[-1]}) must be divisible by num heads"
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, x.shape[-1] // self.num_heads).transpose(1, 2)

    def merge_multi_head(
        self, x: Float[Tensor, "batch_size num_heads sequence_length heads_dim"]
    ) -> Float[Tensor, "batch_size sequence_length heads_dim * num_heads"]:
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], self.num_heads * x.shape[-1])


class Attention(Chain):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 1,
        key_embedding_dim: int | None = None,
        value_embedding_dim: int | None = None,
        inner_dim: int | None = None,
        use_bias: bool = True,
        is_causal: bool | None = None,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        assert (
            embedding_dim % num_heads == 0
        ), f"embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.heads_dim = embedding_dim // num_heads
        self.key_embedding_dim = key_embedding_dim or embedding_dim
        self.value_embedding_dim = value_embedding_dim or embedding_dim
        self.inner_dim = inner_dim or embedding_dim
        self.use_bias = use_bias
        self.is_causal = is_causal
        self.is_optimized = is_optimized
        super().__init__(
            Distribute(
                Linear(
                    in_features=self.embedding_dim,
                    out_features=self.inner_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
                Linear(
                    in_features=self.key_embedding_dim,
                    out_features=self.inner_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
                Linear(
                    in_features=self.value_embedding_dim,
                    out_features=self.inner_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
            ),
            ScaledDotProductAttention(num_heads=num_heads, is_causal=is_causal, is_optimized=is_optimized),
            Linear(
                in_features=self.inner_dim,
                out_features=self.embedding_dim,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )


class SelfAttention(Attention):
    def __init__(
        self,
        embedding_dim: int,
        inner_dim: int | None = None,
        num_heads: int = 1,
        use_bias: bool = True,
        is_causal: bool | None = None,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            inner_dim=inner_dim,
            num_heads=num_heads,
            use_bias=use_bias,
            is_causal=is_causal,
            is_optimized=is_optimized,
            device=device,
            dtype=dtype,
        )
        self.insert(0, Parallel(Identity(), Identity(), Identity()))


class SelfAttention2d(SelfAttention):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        use_bias: bool = True,
        is_causal: bool | None = None,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        self.channels = channels
        super().__init__(
            embedding_dim=channels,
            num_heads=num_heads,
            use_bias=use_bias,
            is_causal=is_causal,
            is_optimized=is_optimized,
            device=device,
            dtype=dtype,
        )
        self.insert(0, Lambda(self.tensor_2d_to_sequence))
        self.append(Lambda(self.sequence_to_tensor_2d))

    def init_context(self) -> Contexts:
        return {"reshape": {"height": None, "width": None}}

    def tensor_2d_to_sequence(
        self, x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch height*width channels"]:
        height, width = x.shape[-2:]
        self.set_context(context="reshape", value={"height": height, "width": width})
        return x.reshape(x.shape[0], x.shape[1], height * width).transpose(1, 2)

    def sequence_to_tensor_2d(
        self, x: Float[Tensor, "batch sequence_length channels"]
    ) -> Float[Tensor, "batch channels height width"]:
        height, width = self.use_context("reshape").values()
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)

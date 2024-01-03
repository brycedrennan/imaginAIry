from torch import device as Device, dtype as DType

import imaginairy.vendored.refiners.fluxion.layers as fl


class CrossAttention(fl.Attention):
    def __init__(
        self,
        embedding_dim: int,
        cross_embedding_dim: int | None = None,
        num_heads: int = 1,
        inner_dim: int | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            key_embedding_dim=cross_embedding_dim,
            num_heads=num_heads,
            inner_dim=inner_dim,
            is_optimized=False,
            device=device,
            dtype=dtype,
        )
        self.cross_embedding_dim = cross_embedding_dim or embedding_dim
        self.insert(index=0, module=fl.Parallel(fl.GetArg(index=0), fl.GetArg(index=1), fl.GetArg(index=1)))


class FeedForward(fl.Residual):
    def __init__(
        self, embedding_dim: int, feed_forward_dim: int, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        self.embedding_dim = embedding_dim
        self.feed_forward_dim = feed_forward_dim
        super().__init__(
            fl.Linear(in_features=embedding_dim, out_features=feed_forward_dim, device=device, dtype=dtype),
            fl.ReLU(),
            fl.Linear(in_features=feed_forward_dim, out_features=embedding_dim, device=device, dtype=dtype),
        )


class SparseSelfAttention(fl.Residual):
    def __init__(
        self,
        embedding_dim: int,
        inner_dim: int | None = None,
        num_heads: int = 1,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        add_sparse_embedding = fl.Residual(fl.UseContext(context="mask_decoder", key="sparse_embedding"))
        super().__init__(
            fl.Parallel(add_sparse_embedding, add_sparse_embedding, fl.Identity()),
            fl.Attention(
                embedding_dim=embedding_dim,
                inner_dim=inner_dim,
                num_heads=num_heads,
                is_optimized=False,
                device=device,
                dtype=dtype,
            ),
        )


class SparseCrossDenseAttention(fl.Residual):
    def __init__(
        self, embedding_dim: int, num_heads: int = 8, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        super().__init__(
            fl.Parallel(
                fl.Residual(
                    fl.UseContext(context="mask_decoder", key="sparse_embedding"),
                ),
                fl.Sum(
                    fl.UseContext(context="mask_decoder", key="dense_embedding"),
                    fl.UseContext(context="mask_decoder", key="dense_positional_embedding"),
                ),
                fl.UseContext(context="mask_decoder", key="dense_embedding"),
            ),
            fl.Attention(
                embedding_dim=embedding_dim,
                inner_dim=embedding_dim // 2,
                num_heads=num_heads,
                is_optimized=False,
                device=device,
                dtype=dtype,
            ),
        )


class DenseCrossSparseAttention(fl.Chain):
    def __init__(
        self, embedding_dim: int, num_heads: int = 8, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        super().__init__(
            fl.Parallel(
                fl.Sum(
                    fl.UseContext(context="mask_decoder", key="dense_embedding"),
                    fl.UseContext(context="mask_decoder", key="dense_positional_embedding"),
                ),
                fl.Residual(
                    fl.UseContext(context="mask_decoder", key="sparse_embedding"),
                ),
                fl.Identity(),
            ),
            fl.Attention(
                embedding_dim=embedding_dim,
                inner_dim=embedding_dim // 2,
                num_heads=num_heads,
                is_optimized=False,
                device=device,
                dtype=dtype,
            ),
        )


class TwoWayTranformerLayer(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        feed_forward_dim: int = 2048,
        use_residual_self_attention: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim

        self_attention = (
            SparseSelfAttention(embedding_dim=embedding_dim, num_heads=num_heads, device=device, dtype=dtype)
            if use_residual_self_attention
            else fl.SelfAttention(
                embedding_dim=embedding_dim, num_heads=num_heads, is_optimized=False, device=device, dtype=dtype
            )
        )

        super().__init__(
            self_attention,
            fl.LayerNorm(normalized_shape=embedding_dim, device=device, dtype=dtype),
            SparseCrossDenseAttention(embedding_dim=embedding_dim, num_heads=num_heads, device=device, dtype=dtype),
            fl.LayerNorm(normalized_shape=embedding_dim, device=device, dtype=dtype),
            FeedForward(embedding_dim=embedding_dim, feed_forward_dim=feed_forward_dim, device=device, dtype=dtype),
            fl.LayerNorm(normalized_shape=embedding_dim, device=device, dtype=dtype),
            fl.Passthrough(
                fl.Sum(
                    fl.UseContext(context="mask_decoder", key="dense_embedding"),
                    DenseCrossSparseAttention(
                        embedding_dim=embedding_dim, num_heads=num_heads, device=device, dtype=dtype
                    ),
                ),
                fl.LayerNorm(normalized_shape=embedding_dim, device=device, dtype=dtype),
                fl.SetContext(context="mask_decoder", key="dense_embedding"),
            ),
        )

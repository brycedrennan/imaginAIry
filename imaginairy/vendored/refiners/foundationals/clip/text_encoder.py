from torch import device as Device, dtype as DType

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.foundationals.clip.common import FeedForward, PositionalEncoder
from imaginairy.vendored.refiners.foundationals.clip.tokenizer import CLIPTokenizer


class TokenEncoder(fl.Embedding):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        super().__init__(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )


class TransformerLayer(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        num_attention_heads: int = 1,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.layer_norm_eps = layer_norm_eps
        super().__init__(
            fl.Residual(
                fl.LayerNorm(
                    normalized_shape=embedding_dim,
                    eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                ),
                fl.SelfAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    is_causal=True,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Residual(
                fl.LayerNorm(
                    normalized_shape=embedding_dim,
                    eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                ),
                FeedForward(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class CLIPTextEncoder(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 768,
        max_sequence_length: int = 77,
        vocabulary_size: int = 49408,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        feedforward_dim: int = 3072,
        layer_norm_eps: float = 1e-5,
        use_quick_gelu: bool = False,
        tokenizer: CLIPTokenizer | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.layer_norm_eps = layer_norm_eps
        self.use_quick_gelu = use_quick_gelu
        super().__init__(
            tokenizer or CLIPTokenizer(sequence_length=max_sequence_length),
            fl.Converter(set_dtype=False),
            fl.Sum(
                TokenEncoder(
                    vocabulary_size=vocabulary_size,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                PositionalEncoder(
                    max_sequence_length=max_sequence_length,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
            *(
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    num_attention_heads=num_attention_heads,
                    feedforward_dim=feedforward_dim,
                    layer_norm_eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
        )
        if use_quick_gelu:
            for gelu, parent in self.walk(predicate=lambda m, _: isinstance(m, fl.GeLU)):
                parent.replace(old_module=gelu, new_module=fl.ApproximateGeLU())


class CLIPTextEncoderL(CLIPTextEncoder):
    """
    CLIPTextEncoderL is the CLIP text encoder with the following parameters:
    embedding_dim=768
    num_layers=12
    num_attention_heads=12
    feedforward_dim=3072
    use_quick_gelu=True

    We replace the GeLU activation function with an approximate GeLU to comply with the original CLIP implementation
    of OpenAI (https://github.com/openai/CLIP/blob/main/clip/model.py#L166)
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            embedding_dim=768,
            num_layers=12,
            num_attention_heads=12,
            feedforward_dim=3072,
            use_quick_gelu=True,
            device=device,
            dtype=dtype,
        )


class CLIPTextEncoderH(CLIPTextEncoder):
    """
    CLIPTextEncoderH is the CLIP text encoder with the following parameters:
    embedding_dim=1024
    num_layers=23
    num_attention_heads=16
    feedforward_dim=4096
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            embedding_dim=1024,
            num_layers=23,
            num_attention_heads=16,
            feedforward_dim=4096,
            device=device,
            dtype=dtype,
        )


class CLIPTextEncoderG(CLIPTextEncoder):
    """
    CLIPTextEncoderG is the CLIP text encoder with the following parameters:
    embedding_dim=1280
    num_layers=32
    num_attention_heads=16
    feedforward_dim=5120
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        tokenizer = CLIPTokenizer(pad_token_id=0)
        super().__init__(
            embedding_dim=1280,
            num_layers=32,
            num_attention_heads=20,
            feedforward_dim=5120,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )

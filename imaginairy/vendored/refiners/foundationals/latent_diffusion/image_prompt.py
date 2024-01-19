import math
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from jaxtyping import Float
from PIL import Image
from torch import Tensor, cat, device as Device, dtype as DType, nn, softmax, zeros_like

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter
from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.fluxion.layers.attentions import ScaledDotProductAttention
from imaginairy.vendored.refiners.fluxion.utils import image_to_tensor, normalize
from imaginairy.vendored.refiners.foundationals.clip.image_encoder import CLIPImageEncoderH

if TYPE_CHECKING:
    from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
    from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TIPAdapter = TypeVar("TIPAdapter", bound="IPAdapter[Any]")  # Self (see PEP 673)


class ImageProjection(fl.Chain):
    def __init__(
        self,
        clip_image_embedding_dim: int = 1024,
        clip_text_embedding_dim: int = 768,
        num_tokens: int = 4,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.clip_image_embedding_dim = clip_image_embedding_dim
        self.clip_text_embedding_dim = clip_text_embedding_dim
        self.num_tokens = num_tokens
        super().__init__(
            fl.Linear(
                in_features=clip_image_embedding_dim,
                out_features=clip_text_embedding_dim * num_tokens,
                device=device,
                dtype=dtype,
            ),
            fl.Reshape(num_tokens, clip_text_embedding_dim),
            fl.LayerNorm(normalized_shape=clip_text_embedding_dim, device=device, dtype=dtype),
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
                bias=False,
                device=device,
                dtype=dtype,
            ),
            fl.GeLU(),
            fl.Linear(
                in_features=self.feedforward_dim,
                out_features=self.embedding_dim,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )


# Adapted from https://github.com/tencent-ailab/IP-Adapter/blob/6212981/ip_adapter/resampler.py
# See also:
# - https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# - https://github.com/lucidrains/flamingo-pytorch
class PerceiverScaledDotProductAttention(fl.Module):
    def __init__(self, head_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        # See https://github.com/tencent-ailab/IP-Adapter/blob/6212981/ip_adapter/resampler.py#L69
        # -> "More stable with f16 than dividing afterwards"
        self.scale = 1 / math.sqrt(math.sqrt(head_dim))

    def forward(
        self,
        key_value: Float[Tensor, "batch sequence_length 2*head_dim*num_heads"],
        query: Float[Tensor, "batch num_tokens head_dim*num_heads"],
    ) -> Float[Tensor, "batch num_tokens head_dim*num_heads"]:
        bs, length, _ = query.shape
        key, value = key_value.chunk(2, dim=-1)

        q = self.reshape_tensor(query)
        k = self.reshape_tensor(key)
        v = self.reshape_tensor(value)

        attention = (q * self.scale) @ (k * self.scale).transpose(-2, -1)
        attention = softmax(input=attention.float(), dim=-1).type(attention.dtype)
        attention = attention @ v

        return attention.permute(0, 2, 1, 3).reshape(bs, length, -1)

    def reshape_tensor(
        self, x: Float[Tensor, "batch length head_dim*num_heads"]
    ) -> Float[Tensor, "batch num_heads length head_dim"]:
        bs, length, _ = x.shape
        x = x.view(bs, length, self.num_heads, -1)
        x = x.transpose(1, 2)
        x = x.reshape(bs, self.num_heads, length, -1)
        return x


class PerceiverAttention(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        super().__init__(
            fl.Distribute(
                fl.LayerNorm(normalized_shape=self.embedding_dim, device=device, dtype=dtype),
                fl.LayerNorm(normalized_shape=self.embedding_dim, device=device, dtype=dtype),
            ),
            fl.Parallel(
                fl.Chain(
                    fl.Lambda(func=self.to_kv),
                    fl.Linear(
                        in_features=self.embedding_dim,
                        out_features=2 * self.inner_dim,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),  # Wkv
                ),
                fl.Chain(
                    fl.GetArg(index=1),
                    fl.Linear(
                        in_features=self.embedding_dim,
                        out_features=self.inner_dim,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),  # Wq
                ),
            ),
            PerceiverScaledDotProductAttention(head_dim=head_dim, num_heads=num_heads),
            fl.Linear(
                in_features=self.inner_dim, out_features=self.embedding_dim, bias=False, device=device, dtype=dtype
            ),
        )

    def to_kv(self, x: Tensor, latents: Tensor) -> Tensor:
        return cat((x, latents), dim=-2)


class LatentsToken(fl.Chain):
    def __init__(
        self, num_tokens: int, latents_dim: int, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        self.num_tokens = num_tokens
        self.latents_dim = latents_dim
        super().__init__(fl.Parameter(num_tokens, latents_dim, device=device, dtype=dtype))


class Transformer(fl.Chain):
    pass


class TransformerLayer(fl.Chain):
    pass


class PerceiverResampler(fl.Chain):
    def __init__(
        self,
        latents_dim: int = 1024,
        num_attention_layers: int = 8,
        num_attention_heads: int = 16,
        head_dim: int = 64,
        num_tokens: int = 8,
        input_dim: int = 768,
        output_dim: int = 1024,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.latents_dim = latents_dim
        self.num_attention_layers = num_attention_layers
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feedforward_dim = 4 * self.latents_dim
        super().__init__(
            fl.Linear(in_features=input_dim, out_features=latents_dim, device=device, dtype=dtype),
            fl.SetContext(context="perceiver_resampler", key="x"),
            LatentsToken(num_tokens, latents_dim, device=device, dtype=dtype),
            Transformer(
                TransformerLayer(
                    fl.Residual(
                        fl.Parallel(fl.UseContext(context="perceiver_resampler", key="x"), fl.Identity()),
                        PerceiverAttention(
                            embedding_dim=latents_dim,
                            head_dim=head_dim,
                            num_heads=num_attention_heads,
                            device=device,
                            dtype=dtype,
                        ),
                    ),
                    fl.Residual(
                        fl.LayerNorm(normalized_shape=latents_dim, device=device, dtype=dtype),
                        FeedForward(
                            embedding_dim=latents_dim, feedforward_dim=self.feedforward_dim, device=device, dtype=dtype
                        ),
                    ),
                )
                for _ in range(num_attention_layers)
            ),
            fl.Linear(in_features=latents_dim, out_features=output_dim, device=device, dtype=dtype),
            fl.LayerNorm(normalized_shape=output_dim, device=device, dtype=dtype),
        )

    def init_context(self) -> Contexts:
        return {"perceiver_resampler": {"x": None}}


class ImageCrossAttention(fl.Chain):
    def __init__(self, text_cross_attention: fl.Attention, scale: float = 1.0) -> None:
        self._scale = scale
        super().__init__(
            fl.Distribute(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="clip_image_embedding"),
                    fl.Linear(
                        in_features=text_cross_attention.key_embedding_dim,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                ),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="clip_image_embedding"),
                    fl.Linear(
                        in_features=text_cross_attention.value_embedding_dim,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                ),
            ),
            ScaledDotProductAttention(
                num_heads=text_cross_attention.num_heads, is_causal=text_cross_attention.is_causal
            ),
            fl.Multiply(self.scale),
        )

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.ensure_find(fl.Multiply).scale = value


class CrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    def __init__(
        self,
        target: fl.Attention,
        scale: float = 1.0,
    ) -> None:
        self._scale = scale
        with self.setup_adapter(target):
            clone = target.structural_copy()
            scaled_dot_product = clone.ensure_find(ScaledDotProductAttention)
            image_cross_attention = ImageCrossAttention(
                text_cross_attention=clone,
                scale=self.scale,
            )
            clone.replace(
                old_module=scaled_dot_product,
                new_module=fl.Sum(
                    scaled_dot_product,
                    image_cross_attention,
                ),
            )
            super().__init__(
                clone,
            )

    @property
    def image_cross_attention(self) -> ImageCrossAttention:
        return self.ensure_find(ImageCrossAttention)

    @property
    def image_key_projection(self) -> fl.Linear:
        return self.image_cross_attention.Distribute[1].Linear

    @property
    def image_value_projection(self) -> fl.Linear:
        return self.image_cross_attention.Distribute[2].Linear

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.image_cross_attention.scale = value

    def load_weights(self, key_tensor: Tensor, value_tensor: Tensor) -> None:
        self.image_key_projection.weight = nn.Parameter(key_tensor)
        self.image_value_projection.weight = nn.Parameter(value_tensor)
        self.image_cross_attention.to(self.device, self.dtype)


class IPAdapter(Generic[T], fl.Chain, Adapter[T]):
    # Prevent PyTorch module registration
    _clip_image_encoder: list[CLIPImageEncoderH]
    _grid_image_encoder: list[CLIPImageEncoderH]
    _image_proj: list[fl.Module]

    def __init__(
        self,
        target: T,
        clip_image_encoder: CLIPImageEncoderH,
        image_proj: fl.Module,
        scale: float = 1.0,
        fine_grained: bool = False,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self.fine_grained = fine_grained
        self._clip_image_encoder = [clip_image_encoder]
        if fine_grained:
            self._grid_image_encoder = [self.convert_to_grid_features(clip_image_encoder)]
        self._image_proj = [image_proj]

        self.sub_adapters = [
            CrossAttentionAdapter(target=cross_attn, scale=scale)
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]

        if weights is not None:
            image_proj_state_dict: dict[str, Tensor] = {
                k.removeprefix("image_proj."): v for k, v in weights.items() if k.startswith("image_proj.")
            }
            self.image_proj.load_state_dict(image_proj_state_dict)

            for i, cross_attn in enumerate(self.sub_adapters):
                cross_attention_weights: list[Tensor] = []
                for k, v in weights.items():
                    prefix = f"ip_adapter.{i:03d}."
                    if not k.startswith(prefix):
                        continue
                    cross_attention_weights.append(v)

                assert len(cross_attention_weights) == 2
                cross_attn.load_weights(*cross_attention_weights)

    @property
    def clip_image_encoder(self) -> CLIPImageEncoderH:
        return self._clip_image_encoder[0]

    @property
    def grid_image_encoder(self) -> CLIPImageEncoderH:
        assert hasattr(self, "_grid_image_encoder")
        return self._grid_image_encoder[0]

    @property
    def image_proj(self) -> fl.Module:
        return self._image_proj[0]

    def inject(self: "TIPAdapter", parent: fl.Chain | None = None) -> "TIPAdapter":
        for adapter in self.sub_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        for adapter in self.sub_adapters:
            adapter.eject()
        super().eject()

    @property
    def scale(self) -> float:
        return self.sub_adapters[0].scale

    @scale.setter
    def scale(self, value: float) -> None:
        for cross_attn in self.sub_adapters:
            cross_attn.scale = value

    def set_scale(self, scale: float) -> None:
        for cross_attn in self.sub_adapters:
            cross_attn.scale = scale

    def set_clip_image_embedding(self, image_embedding: Tensor) -> None:
        self.set_context("ip_adapter", {"clip_image_embedding": image_embedding})

    # These should be concatenated to the CLIP text embedding before setting the UNet context
    def compute_clip_image_embedding(self, image_prompt: Tensor) -> Tensor:
        image_encoder = self.clip_image_encoder if not self.fine_grained else self.grid_image_encoder
        clip_embedding = image_encoder(image_prompt)
        conditional_embedding = self.image_proj(clip_embedding)
        if not self.fine_grained:
            negative_embedding = self.image_proj(zeros_like(clip_embedding))
        else:
            # See https://github.com/tencent-ailab/IP-Adapter/blob/d580c50/tutorial_train_plus.py#L351-L352
            clip_embedding = image_encoder(zeros_like(image_prompt))
            negative_embedding = self.image_proj(clip_embedding)
        return cat((negative_embedding, conditional_embedding))

    def preprocess_image(
        self,
        image: Image.Image,
        size: tuple[int, int] = (224, 224),
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> Tensor:
        # Default mean and std are parameters from https://github.com/openai/CLIP
        return normalize(
            image_to_tensor(image.resize(size), device=self.target.device, dtype=self.target.dtype),
            mean=[0.48145466, 0.4578275, 0.40821073] if mean is None else mean,
            std=[0.26862954, 0.26130258, 0.27577711] if std is None else std,
        )

    @staticmethod
    def convert_to_grid_features(clip_image_encoder: CLIPImageEncoderH) -> CLIPImageEncoderH:
        encoder_clone = clip_image_encoder.structural_copy()
        assert isinstance(encoder_clone[-1], fl.Linear)  # final proj
        assert isinstance(encoder_clone[-2], fl.LayerNorm)  # final normalization
        assert isinstance(encoder_clone[-3], fl.Lambda)  # pooling (classif token)
        for _ in range(3):
            encoder_clone.pop()
        transfomer_layers = encoder_clone[-1]
        assert isinstance(transfomer_layers, fl.Chain) and len(transfomer_layers) == 32
        transfomer_layers.pop()
        return encoder_clone

import math
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch
from jaxtyping import Float
from torch import Size, Tensor

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter
from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.fluxion.utils import gaussian_blur, interpolate
from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.scheduler import Scheduler

if TYPE_CHECKING:
    from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
    from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TSAGAdapter = TypeVar("TSAGAdapter", bound="SAGAdapter[Any]")  # Self (see PEP 673)


class SelfAttentionMap(fl.Passthrough):
    def __init__(self, num_heads: int, context_key: str) -> None:
        self.num_heads = num_heads
        self.context_key = context_key
        super().__init__(
            fl.Lambda(func=self.compute_attention_scores),
            fl.SetContext(context="self_attention_map", key=context_key),
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

    def compute_attention_scores(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        query, key = self.split_to_multi_head(query), self.split_to_multi_head(key)
        _, _, _, dim = query.shape
        attention = query @ key.permute(0, 1, 3, 2)
        attention = attention / math.sqrt(dim)
        return torch.softmax(input=attention, dim=-1)


class SelfAttentionShape(fl.Passthrough):
    def __init__(self, context_key: str) -> None:
        self.context_key = context_key
        super().__init__(
            fl.SetContext(context="self_attention_map", key=context_key, callback=self.register_shape),
        )

    def register_shape(self, shapes: list[Size], x: Tensor) -> None:
        assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}D with shape {x.shape}"
        shapes.append(x.shape[-2:])


class SAGAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(self, target: T, scale: float = 1.0, kernel_size: int = 9, sigma: float = 1.0) -> None:
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma = sigma
        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self: "TSAGAdapter", parent: fl.Chain | None = None) -> "TSAGAdapter":
        return super().inject(parent)

    def eject(self) -> None:
        super().eject()

    def compute_sag_mask(
        self, latents: Float[Tensor, "batch_size channels height width"], classifier_free_guidance: bool = True
    ) -> Float[Tensor, "batch_size channels height width"]:
        attn_map = self.use_context("self_attention_map")["middle_block_attn_map"]
        if classifier_free_guidance:
            unconditional_attn, _ = attn_map.chunk(2)
            attn_map = unconditional_attn
        attn_shape = self.use_context("self_attention_map")["middle_block_attn_shape"].pop()
        assert len(attn_shape) == 2
        b, c, h, w = latents.shape
        attn_h, attn_w = attn_shape
        attn_mask = attn_map.mean(dim=1, keepdim=False).sum(dim=1, keepdim=False) > 1.0
        attn_mask = attn_mask.reshape(b, attn_h, attn_w).unsqueeze(1).repeat(1, c, 1, 1).type(attn_map.dtype)
        return interpolate(attn_mask, Size((h, w)))

    def compute_degraded_latents(
        self, scheduler: Scheduler, latents: Tensor, noise: Tensor, step: int, classifier_free_guidance: bool = True
    ) -> Tensor:
        sag_mask = self.compute_sag_mask(latents=latents, classifier_free_guidance=classifier_free_guidance)
        original_latents = scheduler.remove_noise(x=latents, noise=noise, step=step)
        degraded_latents = gaussian_blur(original_latents, kernel_size=self.kernel_size, sigma=self.sigma)
        degraded_latents = degraded_latents * sag_mask + original_latents * (1 - sag_mask)
        return scheduler.add_noise(degraded_latents, noise=noise, step=step)

    def init_context(self) -> Contexts:
        return {"self_attention_map": {"middle_block_attn_map": None, "middle_block_attn_shape": []}}

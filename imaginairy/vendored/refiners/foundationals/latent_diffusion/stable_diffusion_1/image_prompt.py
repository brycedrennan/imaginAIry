from torch import Tensor

from imaginairy.vendored.refiners.foundationals.clip.image_encoder import CLIPImageEncoderH
from imaginairy.vendored.refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from imaginairy.vendored.refiners.foundationals.latent_diffusion.image_prompt import ImageProjection, IPAdapter, PerceiverResampler
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet


class SD1IPAdapter(IPAdapter[SD1UNet]):
    def __init__(
        self,
        target: SD1UNet,
        clip_image_encoder: CLIPImageEncoderH | None = None,
        image_proj: ImageProjection | PerceiverResampler | None = None,
        scale: float = 1.0,
        fine_grained: bool = False,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        clip_image_encoder = clip_image_encoder or CLIPImageEncoderH(device=target.device, dtype=target.dtype)

        if image_proj is None:
            cross_attn_2d = target.ensure_find(CrossAttentionBlock2d)
            image_proj = (
                ImageProjection(
                    clip_image_embedding_dim=clip_image_encoder.output_dim,
                    clip_text_embedding_dim=cross_attn_2d.context_embedding_dim,
                    device=target.device,
                    dtype=target.dtype,
                )
                if not fine_grained
                else PerceiverResampler(
                    latents_dim=cross_attn_2d.context_embedding_dim,
                    num_attention_layers=4,
                    num_attention_heads=12,
                    head_dim=64,
                    num_tokens=16,
                    input_dim=clip_image_encoder.embedding_dim,  # = dim before final projection
                    output_dim=cross_attn_2d.context_embedding_dim,
                    device=target.device,
                    dtype=target.dtype,
                )
            )
        elif fine_grained:
            assert isinstance(image_proj, PerceiverResampler)

        super().__init__(
            target=target,
            clip_image_encoder=clip_image_encoder,
            image_proj=image_proj,
            scale=scale,
            fine_grained=fine_grained,
            weights=weights,
        )

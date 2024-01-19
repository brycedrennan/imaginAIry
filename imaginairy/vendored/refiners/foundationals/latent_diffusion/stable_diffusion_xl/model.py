import torch
from torch import Tensor, device as Device, dtype as DType

from imaginairy.vendored.refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder
from imaginairy.vendored.refiners.foundationals.latent_diffusion.model import LatentDiffusionModel
from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.ddim import DDIM
from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.scheduler import Scheduler
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.self_attention_guidance import SDXLSAGAdapter
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import DoubleTextEncoder
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet


class SDXLAutoencoder(LatentDiffusionAutoencoder):
    encoder_scale: float = 0.13025


class StableDiffusion_XL(LatentDiffusionModel):
    unet: SDXLUNet
    clip_text_encoder: DoubleTextEncoder

    def __init__(
        self,
        unet: SDXLUNet | None = None,
        lda: SDXLAutoencoder | None = None,
        clip_text_encoder: DoubleTextEncoder | None = None,
        scheduler: Scheduler | None = None,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        unet = unet or SDXLUNet(in_channels=4)
        lda = lda or SDXLAutoencoder()
        clip_text_encoder = clip_text_encoder or DoubleTextEncoder()
        scheduler = scheduler or DDIM(num_inference_steps=30)

        super().__init__(
            unet=unet,
            lda=lda,
            clip_text_encoder=clip_text_encoder,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )

    def compute_clip_text_embedding(self, text: str, negative_text: str | None = None) -> tuple[Tensor, Tensor]:
        conditional_embedding, conditional_pooled_embedding = self.clip_text_encoder(text)
        if text == negative_text:
            return torch.cat(tensors=(conditional_embedding, conditional_embedding), dim=0), torch.cat(
                tensors=(conditional_pooled_embedding, conditional_pooled_embedding), dim=0
            )

        # TODO: when negative_text is None, use zero tensor?
        negative_embedding, negative_pooled_embedding = self.clip_text_encoder(negative_text or "")

        return torch.cat(tensors=(negative_embedding, conditional_embedding), dim=0), torch.cat(
            tensors=(negative_pooled_embedding, conditional_pooled_embedding), dim=0
        )

    @property
    def default_time_ids(self) -> Tensor:
        # [original_height, original_width, crop_top, crop_left, target_height, target_width]
        # See https://arxiv.org/abs/2307.01952 > 2.2 Micro-Conditioning
        time_ids = torch.tensor(data=[1024, 1024, 0, 0, 1024, 1024], device=self.device)
        return time_ids.repeat(2, 1)

    def set_unet_context(
        self,
        *,
        timestep: Tensor,
        clip_text_embedding: Tensor,
        pooled_text_embedding: Tensor,
        time_ids: Tensor,
        **_: Tensor,
    ) -> None:
        self.unet.set_timestep(timestep=timestep)
        self.unet.set_clip_text_embedding(clip_text_embedding=clip_text_embedding)
        self.unet.set_pooled_text_embedding(pooled_text_embedding=pooled_text_embedding)
        self.unet.set_time_ids(time_ids=time_ids)

    def forward(
        self,
        x: Tensor,
        step: int,
        *,
        clip_text_embedding: Tensor,
        pooled_text_embedding: Tensor,
        time_ids: Tensor,
        condition_scale: float = 5.0,
        **kwargs: Tensor,
    ) -> Tensor:
        return super().forward(
            x=x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=condition_scale,
            **kwargs,
        )

    def set_self_attention_guidance(self, enable: bool, scale: float = 1.0) -> None:
        if enable:
            if sag := self._find_sag_adapter():
                sag.scale = scale
            else:
                SDXLSAGAdapter(target=self.unet, scale=scale).inject()
        else:
            if sag := self._find_sag_adapter():
                sag.eject()

    def has_self_attention_guidance(self) -> bool:
        return self._find_sag_adapter() is not None

    def _find_sag_adapter(self) -> SDXLSAGAdapter | None:
        for p in self.unet.get_parents():
            if isinstance(p, SDXLSAGAdapter):
                return p
        return None

    def compute_self_attention_guidance(
        self,
        x: Tensor,
        noise: Tensor,
        step: int,
        *,
        clip_text_embedding: Tensor,
        pooled_text_embedding: Tensor,
        time_ids: Tensor,
        **kwargs: Tensor,
    ) -> Tensor:
        sag = self._find_sag_adapter()
        assert sag is not None

        degraded_latents = sag.compute_degraded_latents(
            scheduler=self.scheduler,
            latents=x,
            noise=noise,
            step=step,
            classifier_free_guidance=True,
        )

        negative_text_embedding, _ = clip_text_embedding.chunk(2)
        negative_pooled_embedding, _ = pooled_text_embedding.chunk(2)
        timestep = self.scheduler.timesteps[step].unsqueeze(dim=0)
        time_ids, _ = time_ids.chunk(2)

        self.set_unet_context(
            timestep=timestep,
            clip_text_embedding=negative_text_embedding,
            pooled_text_embedding=negative_pooled_embedding,
            time_ids=time_ids,
        )
        if "ip_adapter" in self.unet.provider.contexts:
            # this implementation is a bit hacky, it should be refactored in the future
            ip_adapter_context = self.unet.use_context("ip_adapter")
            image_embedding_copy = ip_adapter_context["clip_image_embedding"].clone()
            ip_adapter_context["clip_image_embedding"], _ = ip_adapter_context["clip_image_embedding"].chunk(2)
            degraded_noise = self.unet(degraded_latents)
            ip_adapter_context["clip_image_embedding"] = image_embedding_copy
        else:
            degraded_noise = self.unet(degraded_latents)

        return sag.scale * (noise - degraded_noise)

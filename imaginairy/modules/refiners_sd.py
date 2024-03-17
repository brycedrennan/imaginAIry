"""Refinement modules for image generation"""

import logging
import math
from functools import lru_cache
from typing import Any, List, Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor, device as Device, dtype as DType, nn
from torch.nn import functional as F

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy import config
from imaginairy.schema import WeightedPrompt
from imaginairy.utils.downloads import get_cached_url_path
from imaginairy.utils.feather_tile import rebuild_image, tile_image
from imaginairy.vendored.refiners.fluxion.layers.attentions import (
    ScaledDotProductAttention,
)
from imaginairy.vendored.refiners.fluxion.layers.chain import ChainError
from imaginairy.vendored.refiners.fluxion.utils import image_to_tensor, interpolate
from imaginairy.vendored.refiners.foundationals.clip.text_encoder import (
    CLIPTextEncoderL,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion import (
    SD1IPAdapter,
    SDXLIPAdapter,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.model import (
    TLatentDiffusionModel,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.ddim import (
    DDIM,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.scheduler import (
    Scheduler,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.self_attention_guidance import (
    SelfAttentionMap,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.controlnet import (
    Controlnet,
    SD1ControlnetAdapter,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.model import (
    SD1Autoencoder,
    SD1UNet,
    StableDiffusion_1 as RefinerStableDiffusion_1,
    StableDiffusion_1_Inpainting as RefinerStableDiffusion_1_Inpainting,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.model import (
    SDXLAutoencoder,
    StableDiffusion_XL as RefinerStableDiffusion_XL,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import (
    DoubleTextEncoder,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import (
    SDXLUNet,
)
from imaginairy.weight_management.conversion import cast_weights
from imaginairy.weight_management.translators import (
    diffusers_ip_adapter_plus_sd15_to_refiners_translator,
    diffusers_ip_adapter_plus_sdxl_to_refiners_translator,
    diffusers_ip_adapter_sd15_to_refiners_translator,
    diffusers_ip_adapter_sdxl_to_refiners_translator,
    transformers_image_encoder_to_refiners_translator,
)

logger = logging.getLogger(__name__)

TileModeType = Literal["", "x", "y", "xy"]


def _tile_mode_conv2d_conv_forward(
    self,
    input: torch.Tensor,  # noqa
    weight: torch.Tensor,
    bias: torch.Tensor,
):
    if self.padding_mode_x == self.padding_mode_y:
        self.padding_mode = self.padding_mode_x
        return self._orig_conv_forward(input, weight, bias)

    w1 = F.pad(input, self.padding_x, mode=self.padding_modeX)
    del input

    w2 = F.pad(w1, self.padding_y, mode=self.padding_modeY)
    del w1

    return F.conv2d(w2, weight, bias, self.stride, (0, 0), self.dilation, self.groups)


class TileModeMixin(nn.Module):
    def set_tile_mode(self, tile_mode: TileModeType = ""):
        """
        For creating seamless tile images.

        Args:
            tile_mode: One of "", "x", "y", "xy". If "x", the image will be tiled horizontally. If "y", the image will be
                tiled vertically. If "xy", the image will be tiled both horizontally and vertically.
        """
        padding_mode_x = "circular" if "x" in tile_mode else "constant"
        padding_mode_y = "circular" if "y" in tile_mode else "constant"
        for m in self.modules():
            if not isinstance(m, nn.Conv2d):
                continue
            if not hasattr(m, "_orig_conv_forward"):
                # patch with a function that can handle tiling in a single direction
                m._initial_padding_mode = m.padding_mode  # type: ignore
                m._orig_conv_forward = m._conv_forward  # type: ignore
                m._conv_forward = _tile_mode_conv2d_conv_forward.__get__(m, nn.Conv2d)  # type: ignore
            m.padding_mode_x = padding_mode_x  # type: ignore
            m.padding_mode_y = padding_mode_y  # type: ignore
            rprt: list[int] = m._reversed_padding_repeated_twice
            m.padding_x = (rprt[0], rprt[1], 0, 0)  # type: ignore
            m.padding_y = (0, 0, rprt[2], rprt[3])  # type: ignore


class SD1ImagePromptMixin(nn.Module):
    def _get_ip_adapter(self, model_type: str):
        valid_model_types = ["normal", "plus", "plus-face"]
        if model_type not in valid_model_types:
            msg = f"IP Adapter model_type must be one of {valid_model_types}"
            raise ValueError(msg)

        ip_adapter_weights_path = get_cached_url_path(
            config.IP_ADAPTER_WEIGHT_LOCATIONS["sd15"][model_type]
        )
        clip_image_weights_path = get_cached_url_path(config.SD21_UNCLIP_WEIGHTS_URL)
        if "plus" in model_type:
            ip_adapter_weight_translator = (
                diffusers_ip_adapter_plus_sd15_to_refiners_translator()
            )
        else:
            ip_adapter_weight_translator = (
                diffusers_ip_adapter_sd15_to_refiners_translator()
            )
        clip_image_weight_translator = (
            transformers_image_encoder_to_refiners_translator()
        )

        ip_adapter = SD1IPAdapter(
            target=self.unet,
            weights=ip_adapter_weight_translator.load_and_translate_weights(
                ip_adapter_weights_path
            ),
            fine_grained="plus" in model_type,
        )
        ip_adapter.clip_image_encoder.load_state_dict(
            clip_image_weight_translator.load_and_translate_weights(
                clip_image_weights_path
            ),
            assign=True,
        )
        ip_adapter.to(device=self.unet.device, dtype=self.unet.dtype)
        ip_adapter.clip_image_encoder.to(device=self.unet.device, dtype=self.unet.dtype)
        return ip_adapter

    def set_image_prompt(
        self, images: list[Image.Image], scale: float, model_type: str = "normal"
    ):
        ip_adapter = self._get_ip_adapter(model_type)
        ip_adapter.inject()

        ip_adapter.set_scale(scale)
        image_embeddings = []
        for image in images:
            image_embedding = ip_adapter.compute_clip_image_embedding(
                ip_adapter.preprocess_image(image).to(device=self.unet.device)
            )
            image_embeddings.append(image_embedding)

        clip_image_embedding = sum(image_embeddings) / len(image_embeddings)

        ip_adapter.set_clip_image_embedding(clip_image_embedding)


class StableDiffusion_1(TileModeMixin, SD1ImagePromptMixin, RefinerStableDiffusion_1):
    def __init__(
        self,
        unet: SD1UNet | None = None,
        lda: SD1Autoencoder | None = None,
        clip_text_encoder: CLIPTextEncoderL | None = None,
        scheduler: Scheduler | None = None,
        device: Device | str | None = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        unet = unet or SD1UNet(in_channels=4)
        lda = lda or SD1Autoencoder()
        clip_text_encoder = clip_text_encoder or CLIPTextEncoderL()
        scheduler = scheduler or DDIM(num_inference_steps=50)
        fl.Module.__init__(self)

        # all this is to allow us to make structural copies without unnecessary device or dtype shuffeling
        # since default behavior was to put everything on the same device and dtype and we want the option to
        # not alter them from whatever they're already set to
        self.unet = unet
        self.lda = lda
        self.clip_text_encoder = clip_text_encoder
        self.scheduler = scheduler
        to_kwargs: dict[str, Any] = {}

        if device is not None:
            device = device if isinstance(device, Device) else Device(device=device)
            to_kwargs["device"] = device
        if dtype is not None:
            to_kwargs["dtype"] = dtype

        self.device = device  # type: ignore
        self.dtype = dtype

        if to_kwargs:
            self.unet = unet.to(**to_kwargs)
            self.lda = lda.to(**to_kwargs)
            self.clip_text_encoder = clip_text_encoder.to(**to_kwargs)
            self.scheduler = scheduler.to(**to_kwargs)

    def calculate_text_conditioning_kwargs(
        self,
        positive_prompts: List[WeightedPrompt],
        negative_prompts: List[WeightedPrompt],
        positive_conditioning_override: Tensor | None = None,
    ):
        import torch

        from imaginairy.utils.log_utils import log_conditioning

        neutral_conditioning = self.prompts_to_embeddings(negative_prompts)
        log_conditioning(neutral_conditioning, "neutral conditioning")

        if positive_conditioning_override is None:
            positive_conditioning = self.prompts_to_embeddings(positive_prompts)
        else:
            positive_conditioning = positive_conditioning_override
        log_conditioning(positive_conditioning, "positive conditioning")

        clip_text_embedding = torch.cat(
            tensors=(neutral_conditioning, positive_conditioning), dim=0
        )
        return {"clip_text_embedding": clip_text_embedding}

    def prompts_to_embeddings(self, prompts: List[WeightedPrompt]) -> Tensor:
        import torch

        total_weight = sum(wp.weight for wp in prompts)
        if str(self.clip_text_encoder.device) == "cpu":
            self.clip_text_encoder = self.clip_text_encoder.to(dtype=torch.float32)
        conditioning = sum(
            self.clip_text_encoder(wp.text) * (wp.weight / total_weight)
            for wp in prompts
        )

        return conditioning


class SDXLImagePromptMixin(nn.Module):
    def _get_ip_adapter(self, model_type: str):
        valid_model_types = ["normal", "plus", "plus-face"]
        if model_type not in valid_model_types:
            msg = f"IP Adapter model_type must be one of {valid_model_types}"
            raise ValueError(msg)

        ip_adapter_weights_path = get_cached_url_path(
            config.IP_ADAPTER_WEIGHT_LOCATIONS["sdxl"][model_type]
        )
        clip_image_weights_path = get_cached_url_path(config.SD21_UNCLIP_WEIGHTS_URL)
        if "plus" in model_type:
            ip_adapter_weight_translator = (
                diffusers_ip_adapter_plus_sdxl_to_refiners_translator()
            )
        else:
            ip_adapter_weight_translator = (
                diffusers_ip_adapter_sdxl_to_refiners_translator()
            )
        clip_image_weight_translator = (
            transformers_image_encoder_to_refiners_translator()
        )

        ip_adapter = SDXLIPAdapter(
            target=self.unet,
            weights=ip_adapter_weight_translator.load_and_translate_weights(
                ip_adapter_weights_path
            ),
            fine_grained="plus" in model_type,
        )
        ip_adapter.clip_image_encoder.load_state_dict(
            clip_image_weight_translator.load_and_translate_weights(
                clip_image_weights_path
            ),
            assign=True,
        )
        ip_adapter.to(device=self.unet.device, dtype=self.unet.dtype)
        ip_adapter.clip_image_encoder.to(device=self.unet.device, dtype=self.unet.dtype)
        return ip_adapter

    def set_image_prompt(
        self, images: list[Image.Image], scale: float, model_type: str = "normal"
    ):
        ip_adapter = self._get_ip_adapter(model_type)
        ip_adapter.inject()

        ip_adapter.set_scale(scale)
        image_embeddings = []
        for image in images:
            image_embedding = ip_adapter.compute_clip_image_embedding(
                ip_adapter.preprocess_image(image).to(device=self.unet.device)
            )
            image_embeddings.append(image_embedding)

        clip_image_embedding = sum(image_embeddings) / len(image_embeddings)

        ip_adapter.set_clip_image_embedding(clip_image_embedding)


class StableDiffusion_XL(
    TileModeMixin, SDXLImagePromptMixin, RefinerStableDiffusion_XL
):
    def __init__(
        self,
        unet: SDXLUNet | None = None,
        lda: SDXLAutoencoder | None = None,
        clip_text_encoder: DoubleTextEncoder | None = None,
        scheduler: Scheduler | None = None,
        device: Device | str | None = "cpu",
        dtype: DType | None = None,
    ) -> None:
        unet = unet or SDXLUNet(in_channels=4)
        lda = lda or SDXLAutoencoder()
        clip_text_encoder = clip_text_encoder or DoubleTextEncoder()
        scheduler = scheduler or DDIM(num_inference_steps=30)
        fl.Module.__init__(self)

        # all this is to allow us to make structural copies without unnecessary device or dtype shuffeling
        # since default behavior was to put everything on the same device and dtype and we want the option to
        # not alter them from whatever they're already set to
        self.unet = unet
        self.lda = lda
        self.clip_text_encoder = clip_text_encoder
        self.scheduler = scheduler
        to_kwargs: dict[str, Any] = {}

        if device is not None:
            device = device if isinstance(device, Device) else Device(device=device)
            to_kwargs["device"] = device
        if dtype is not None:
            to_kwargs["dtype"] = dtype

        self.device = device  # type: ignore
        self.dtype = dtype  # type: ignore
        self.unet = unet
        self.lda = lda
        self.clip_text_encoder = clip_text_encoder
        self.scheduler = scheduler
        if to_kwargs:
            self.unet = self.unet.to(**to_kwargs)
            self.lda = self.lda.to(**to_kwargs)
            self.clip_text_encoder = self.clip_text_encoder.to(**to_kwargs)
            self.scheduler = self.scheduler.to(**to_kwargs)

    def forward(  # type: ignore
        self,
        x: Tensor,
        step: int,
        *,
        clip_text_embedding: Tensor,
        pooled_text_embedding: Tensor,
        time_ids: Tensor | None = None,
        condition_scale: float = 5.0,
        **kwargs: Tensor,
    ) -> Tensor:
        time_ids = time_ids or self.default_time_ids
        return super().forward(
            x=x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=condition_scale,
            **kwargs,
        )

    def structural_copy(self: TLatentDiffusionModel) -> TLatentDiffusionModel:
        logger.debug("Making structural copy of StableDiffusion_XL model")

        sd = self.__class__(
            unet=self.unet.structural_copy(),
            lda=self.lda.structural_copy(),
            clip_text_encoder=self.clip_text_encoder,
            scheduler=self.scheduler,
            device=self.device,
            dtype=None,  # type: ignore
        )
        logger.debug(
            f"dtype: {sd.dtype} unet-dtype:{sd.unet.dtype} lda-dtype:{sd.lda.dtype} text-encoder-dtype:{sd.clip_text_encoder.dtype} scheduler-dtype:{sd.scheduler.dtype}"
        )
        return sd

    def calculate_text_conditioning_kwargs(
        self,
        positive_prompts: List[WeightedPrompt],
        negative_prompts: List[WeightedPrompt],
        positive_conditioning_override: Tensor | None = None,
    ):
        import torch

        from imaginairy.utils.log_utils import log_conditioning

        (
            neutral_clip_text_embedding,
            neutral_pooled_text_embedding,
        ) = self.prompts_to_embeddings(negative_prompts)
        log_conditioning(neutral_clip_text_embedding, "neutral_clip_text_embedding")
        log_conditioning(neutral_pooled_text_embedding, "neutral_pooled_text_embedding")

        (
            positive_clip_text_embedding,
            positive_pooled_text_embedding,
        ) = self.prompts_to_embeddings(positive_prompts)
        log_conditioning(positive_clip_text_embedding, "positive_clip_text_embedding")
        log_conditioning(
            positive_pooled_text_embedding, "positive_pooled_text_embedding"
        )

        return {
            "clip_text_embedding": torch.cat(
                tensors=(neutral_clip_text_embedding, positive_clip_text_embedding),
                dim=0,
            ),
            "pooled_text_embedding": torch.cat(
                tensors=(neutral_pooled_text_embedding, positive_pooled_text_embedding),
                dim=0,
            ),
        }

    def prompts_to_embeddings(
        self, prompts: List[WeightedPrompt]
    ) -> tuple[Tensor, Tensor]:
        import torch

        total_weight = sum(wp.weight for wp in prompts)
        if str(self.clip_text_encoder.device) == "cpu":
            self.clip_text_encoder = self.clip_text_encoder.to(dtype=torch.float32)

        embeddings = [self.clip_text_encoder(wp.text) for wp in prompts]
        clip_text_embedding = (
            sum(emb[0] * wp.weight for emb, wp in zip(embeddings, prompts))
            / total_weight
        )
        pooled_text_embedding = (
            sum(emb[1] * wp.weight for emb, wp in zip(embeddings, prompts))
            / total_weight
        )

        return clip_text_embedding, pooled_text_embedding  # type: ignore


class StableDiffusion_1_Inpainting(
    TileModeMixin, SD1ImagePromptMixin, RefinerStableDiffusion_1_Inpainting
):
    def compute_self_attention_guidance(
        self,
        x: Tensor,
        noise: Tensor,
        step: int,
        *,
        clip_text_embedding: Tensor,
        **kwargs: Tensor,
    ) -> Tensor:
        sag = self._find_sag_adapter()
        assert sag is not None
        assert self.mask_latents is not None
        assert self.target_image_latents is not None

        degraded_latents = sag.compute_degraded_latents(
            scheduler=self.scheduler,
            latents=x,
            noise=noise,
            step=step,
            classifier_free_guidance=True,
        )

        negative_embedding, _ = clip_text_embedding.chunk(2)
        timestep = self.scheduler.timesteps[step].unsqueeze(dim=0)
        self.set_unet_context(
            timestep=timestep, clip_text_embedding=negative_embedding, **kwargs
        )
        x = torch.cat(
            tensors=(degraded_latents, self.mask_latents, self.target_image_latents),
            dim=1,
        )
        if "ip_adapter" in self.unet.provider.contexts:
            # this implementation is a bit hacky, it should be refactored in the future
            ip_adapter_context = self.unet.use_context("ip_adapter")
            image_embedding_copy = ip_adapter_context["clip_image_embedding"].clone()
            ip_adapter_context["clip_image_embedding"], _ = ip_adapter_context[
                "clip_image_embedding"
            ].chunk(2)
            degraded_noise = self.unet(x)
            ip_adapter_context["clip_image_embedding"] = image_embedding_copy
        else:
            degraded_noise = self.unet(x)

        return sag.scale * (noise - degraded_noise)

    def calculate_text_conditioning_kwargs(
        self,
        positive_prompts: List[WeightedPrompt],
        negative_prompts: List[WeightedPrompt],
        positive_conditioning_override: Tensor | None = None,
    ):
        import torch

        from imaginairy.utils.log_utils import log_conditioning

        neutral_conditioning = self.prompts_to_embeddings(negative_prompts)
        log_conditioning(neutral_conditioning, "neutral conditioning")

        if positive_conditioning_override is None:
            positive_conditioning = self.prompts_to_embeddings(positive_prompts)
        else:
            positive_conditioning = positive_conditioning_override
        log_conditioning(positive_conditioning, "positive conditioning")

        clip_text_embedding = torch.cat(
            tensors=(neutral_conditioning, positive_conditioning), dim=0
        )
        return {"clip_text_embedding": clip_text_embedding}

    def prompts_to_embeddings(self, prompts: List[WeightedPrompt]) -> Tensor:
        import torch

        total_weight = sum(wp.weight for wp in prompts)
        if str(self.clip_text_encoder.device) == "cpu":  # type: ignore
            self.clip_text_encoder = self.clip_text_encoder.to(dtype=torch.float32)  # type: ignore
        conditioning = sum(
            self.clip_text_encoder(wp.text) * (wp.weight / total_weight)
            for wp in prompts
        )

        return conditioning


class StableDiffusion_XL_Inpainting(StableDiffusion_XL):
    def __init__(
        self,
        unet: SDXLUNet | None = None,
        lda: SDXLAutoencoder | None = None,
        clip_text_encoder: DoubleTextEncoder | None = None,
        scheduler: Scheduler | None = None,
        device: Device | str | None = "cpu",
        dtype: DType | None = None,
    ) -> None:
        self.mask_latents: Tensor | None = None
        self.target_image_latents: Tensor | None = None
        super().__init__(
            unet=unet,
            lda=lda,
            clip_text_encoder=clip_text_encoder,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        step: int,
        *,
        clip_text_embedding: Tensor,
        pooled_text_embedding: Tensor,
        time_ids: Tensor | None = None,
        condition_scale: float = 5.0,
        **_: Tensor,
    ) -> Tensor:
        assert self.mask_latents is not None
        assert self.target_image_latents is not None
        x = torch.cat(tensors=(x, self.mask_latents, self.target_image_latents), dim=1)
        return super().forward(
            x=x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=condition_scale,
        )

    def set_inpainting_conditions(
        self,
        target_image: Image.Image,
        mask: Image.Image,
        latents_size: tuple[int, int] = (64, 64),
    ) -> tuple[Tensor, Tensor]:
        target_image = target_image.convert(mode="RGB")
        mask = mask.convert(mode="L")

        mask_tensor = torch.tensor(
            data=np.array(object=mask).astype(dtype=np.float32) / 255.0
        ).to(device=self.device)
        mask_tensor = (
            (mask_tensor > 0.5)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .to(dtype=self.unet.dtype)
        )

        self.mask_latents = interpolate(x=mask_tensor, factor=torch.Size(latents_size))

        init_image_tensor = (
            image_to_tensor(
                image=target_image, device=self.device, dtype=self.unet.dtype
            )
            * 2
            - 1
        )
        masked_init_image = init_image_tensor * (1 - mask_tensor)
        self.target_image_latents = self.lda.encode(
            x=masked_init_image.to(dtype=self.lda.dtype)
        )
        assert self.target_image_latents is not None
        self.target_image_latents = self.target_image_latents.to(dtype=self.unet.dtype)

        return self.mask_latents, self.target_image_latents  # type: ignore

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
        assert self.mask_latents is not None
        assert self.target_image_latents is not None

        degraded_latents = sag.compute_degraded_latents(
            scheduler=self.scheduler,
            latents=x,
            noise=noise,
            step=step,
            classifier_free_guidance=True,
        )

        negative_embedding, _ = clip_text_embedding.chunk(2)
        negative_pooled_embedding, _ = pooled_text_embedding.chunk(2)
        timestep = self.scheduler.timesteps[step].unsqueeze(dim=0)
        time_ids, _ = time_ids.chunk(2)
        self.set_unet_context(
            timestep=timestep,
            clip_text_embedding=negative_embedding,
            pooled_text_embedding=negative_pooled_embedding,
            time_ids=time_ids,
            **kwargs,
        )
        x = torch.cat(
            tensors=(degraded_latents, self.mask_latents, self.target_image_latents),
            dim=1,
        )
        if "ip_adapter" in self.unet.provider.contexts:
            # this implementation is a bit hacky, it should be refactored in the future
            ip_adapter_context = self.unet.use_context("ip_adapter")
            image_embedding_copy = ip_adapter_context["clip_image_embedding"].clone()
            ip_adapter_context["clip_image_embedding"], _ = ip_adapter_context[
                "clip_image_embedding"
            ].chunk(2)
            degraded_noise = self.unet(x)
            ip_adapter_context["clip_image_embedding"] = image_embedding_copy
        else:
            degraded_noise = self.unet(x)

        return sag.scale * (noise - degraded_noise)


class SlicedEncoderMixin(nn.Module):
    max_chunk_size = 2048
    min_chunk_size = 32

    def encode(self, x: Tensor) -> Tensor:
        return self.sliced_encode(x)

    def sliced_encode(self, x: Tensor, chunk_size: int = 128 * 8) -> Tensor:
        """
        Encodes the image in slices (for lower memory usage).
        """
        b, c, h, w = x.size()
        final_tensor = torch.zeros(
            [1, 4, math.floor(h / 8), math.floor(w / 8)], device=x.device
        )
        overlap_pct = 0.5
        encoder = self[0]  # type: ignore
        for x_img in x.split(1):
            chunks = tile_image(
                x_img, tile_size=chunk_size, overlap_percent=overlap_pct
            )
            encoded_chunks = [encoder(ic) * self.encoder_scale for ic in chunks]

            final_tensor = rebuild_image(
                encoded_chunks,
                base_img=final_tensor,
                tile_size=chunk_size // 8,
                overlap_percent=overlap_pct,
            )

        return final_tensor

    def decode(self, x):
        while self.__class__.max_chunk_size > self.__class__.min_chunk_size:
            if self.max_chunk_size**2 > x.shape[2] * x.shape[3]:
                try:
                    return self.decode_all_at_once(x)
                except ChainError as e:
                    if "OutOfMemoryError" not in str(e):
                        raise
                    new_size = int(math.sqrt(x.shape[2] * x.shape[3])) // 2
                    # make sure it's an even number
                    new_size = new_size - (new_size % 2)
                    self.__class__.max_chunk_size = new_size
                    logger.info(
                        f"Ran out of memory. Trying tiled decode with chunk size {self.__class__.max_chunk_size}"
                    )
            else:
                try:
                    return self.decode_sliced(x, chunk_size=self.max_chunk_size)
                except ChainError as e:
                    if "OutOfMemoryError" not in str(e):
                        raise
                    self.__class__.max_chunk_size = self.max_chunk_size // 2
                    # make sure it's an even number
                    self.__class__.max_chunk_size -= self.__class__.max_chunk_size % 2
                    self.__class__.max_chunk_size = max(
                        self.__class__.max_chunk_size, self.__class__.min_chunk_size
                    )
                    logger.info(
                        f"Ran out of memory. Trying tiled decode with chunk size {self.__class__.max_chunk_size}"
                    )
        raise RuntimeError("Could not decode image")

    def decode_all_at_once(self, x: Tensor) -> Tensor:
        decoder = self[1]  # type: ignore
        x = decoder(x / self.encoder_scale)
        return x

    def decode_sliced(self, x, chunk_size=128):
        """
        decodes the tensor in slices.

        This results in image portions that don't exactly match, so we overlap, feather, and merge to reduce
        (but not completely eliminate) impact.
        """
        b, c, h, w = x.size()
        final_tensor = torch.zeros([1, 3, h * 8, w * 8], device=x.device)
        for x_latent in x.split(1):
            decoded_chunks = []
            overlap_pct = 0.5
            chunks = tile_image(
                x_latent, tile_size=chunk_size, overlap_percent=overlap_pct
            )

            for latent_chunk in chunks:
                # latent_chunk = self.post_quant_conv(latent_chunk)
                dec = self.decode_all_at_once(latent_chunk)
                decoded_chunks.append(dec)
            final_tensor = rebuild_image(
                decoded_chunks,
                base_img=final_tensor,
                tile_size=chunk_size * 8,
                overlap_percent=overlap_pct,
            )

            return final_tensor


class SD1AutoencoderSliced(SlicedEncoderMixin, SD1Autoencoder):
    pass


class SDXLAutoencoderSliced(SlicedEncoderMixin, SDXLAutoencoder):
    pass


def add_sliced_attention_to_scaled_dot_product_attention(cls):
    """
    Patch refiners ScaledDotProductAttention so that it uses sliced attention

    It reduces peak memory usage.
    """

    def _sliced_attention(self, query, key, value, slice_size, is_causal=None):
        _, num_queries, _ = query.shape
        output = torch.zeros_like(query)
        for start_idx in range(0, num_queries, slice_size):
            end_idx = min(start_idx + slice_size, num_queries)
            output[:, start_idx:end_idx, :] = self._process_attention(
                query[:, start_idx:end_idx, :], key, value, is_causal
            )
        return output

    cls._sliced_attention = _sliced_attention

    def new_forward(self, query, key, value, is_causal=None):
        return self._sliced_attention(
            query, key, value, is_causal=is_causal, slice_size=2048
        )

    cls.forward = new_forward

    def _process_attention(self, query, key, value, is_causal=None):
        return self.merge_multi_head(
            x=self.dot_product(
                query=self.split_to_multi_head(query),
                key=self.split_to_multi_head(key),
                value=self.split_to_multi_head(value),
                is_causal=(
                    is_causal
                    if is_causal is not None
                    else (self.is_causal if self.is_causal is not None else False)
                ),
            )
        )

    cls._process_attention = _process_attention
    logger.debug(f"Patched {cls.__name__} with sliced attention")


add_sliced_attention_to_scaled_dot_product_attention(ScaledDotProductAttention)


@lru_cache
def monkeypatch_sd1controlnetadapter():
    """
    Another horrible thing.

    I needed to be able to cache the controlnet objects so I wouldn't be making new ones on every image generation.
    """

    def __init__(
        self,
        target: SD1UNet,
        name: str,
        weights_location: str,
    ) -> None:
        self.name = name
        controlnet = get_controlnet(
            name=name,
            weights_location=weights_location,
            device=target.device,
            dtype=target.dtype,
        )
        logger.debug(
            f"controlnet: {name} loaded to device {target.device} and type {target.dtype}"
        )

        self._controlnet: list[Controlnet] = [  # type: ignore
            controlnet
        ]  # not registered by PyTorch

        with self.setup_adapter(target):
            super(SD1ControlnetAdapter, self).__init__(target)

    SD1ControlnetAdapter.__init__ = __init__


monkeypatch_sd1controlnetadapter()


@lru_cache
def monkeypatch_self_attention_guidance():
    def new_compute_attention_scores(
        self, query: Tensor, key: Tensor, value: Tensor, slice_size=2048
    ) -> Tensor:
        query, key = self.split_to_multi_head(query), self.split_to_multi_head(key)
        batch_size, num_heads, num_queries, dim = query.shape

        output = torch.zeros(
            batch_size,
            num_heads,
            num_queries,
            num_queries,
            device=query.device,
            dtype=query.dtype,
        )
        for start_idx in range(0, num_queries, slice_size):
            end_idx = min(start_idx + slice_size, num_queries)
            sliced_query = query[:, :, start_idx:end_idx, :]
            attention_slice = sliced_query @ key.permute(0, 1, 3, 2)
            attention_slice = attention_slice / math.sqrt(dim)
            output_slice = torch.softmax(input=attention_slice, dim=-1)
            output[:, :, start_idx:end_idx, :] = output_slice

        return output

    def compute_attention_scores_logged(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tensor:
        print(f"query.shape: {query.shape}")
        print(f"key.shape: {key.shape}")
        query, key = self.split_to_multi_head(query), self.split_to_multi_head(key)
        print(f"mh query.shape: {query.shape}")
        print(f"mh key.shape: {key.shape}")
        _, _, _, dim = query.shape
        attention = query @ key.permute(0, 1, 3, 2)
        attention = attention / math.sqrt(dim)
        print(f"attention shape: {attention.shape}")
        result = torch.softmax(input=attention, dim=-1)
        print(f"result.shape: {result.shape}")
        return result

    SelfAttentionMap.compute_attention_scores = new_compute_attention_scores


monkeypatch_self_attention_guidance()


@lru_cache(maxsize=1)
def get_controlnet(name, weights_location, device, dtype):
    from imaginairy.utils.model_manager import load_state_dict

    controlnet_state_dict = load_state_dict(weights_location, half_mode=False)
    controlnet_state_dict = cast_weights(
        source_weights=controlnet_state_dict,
        source_model_name="controlnet-1-1",
        source_component_name="all",
        source_format="diffusers",
        dest_format="refiners",
    )

    controlnet = Controlnet(name=name, scale=1, device=device, dtype=dtype)
    controlnet.load_state_dict(controlnet_state_dict)
    return controlnet

from abc import ABC, abstractmethod
from typing import TypeVar

import torch
from PIL import Image
from torch import Tensor, device as Device, dtype as DType

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder
from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.scheduler import Scheduler

T = TypeVar("T", bound="fl.Module")

TLatentDiffusionModel = TypeVar("TLatentDiffusionModel", bound="LatentDiffusionModel")


class LatentDiffusionModel(fl.Module, ABC):
    def __init__(
        self,
        unet: fl.Module,
        lda: LatentDiffusionAutoencoder,
        clip_text_encoder: fl.Module,
        scheduler: Scheduler,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        super().__init__()
        self.device: Device = device if isinstance(device, Device) else Device(device=device)
        self.dtype = dtype
        self.unet = unet.to(device=self.device, dtype=self.dtype)
        self.lda = lda.to(device=self.device, dtype=self.dtype)
        self.clip_text_encoder = clip_text_encoder.to(device=self.device, dtype=self.dtype)
        self.scheduler = scheduler.to(device=self.device, dtype=self.dtype)

    def set_inference_steps(self, num_steps: int, first_step: int = 0) -> None:
        initial_diffusion_rate = self.scheduler.initial_diffusion_rate
        final_diffusion_rate = self.scheduler.final_diffusion_rate
        device, dtype = self.scheduler.device, self.scheduler.dtype
        self.scheduler = self.scheduler.__class__(
            num_inference_steps=num_steps,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
            first_inference_step=first_step,
        ).to(device=device, dtype=dtype)

    def init_latents(
        self,
        size: tuple[int, int],
        init_image: Image.Image | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        height, width = size
        if noise is None:
            noise = torch.randn(1, 4, height // 8, width // 8, device=self.device)
        assert list(noise.shape[2:]) == [
            height // 8,
            width // 8,
        ], f"noise shape is not compatible: {noise.shape}, with size: {size}"
        if init_image is None:
            return noise
        encoded_image = self.lda.encode_image(image=init_image.resize(size=(width, height)))
        return self.scheduler.add_noise(
            x=encoded_image,
            noise=noise,
            step=self.scheduler.first_inference_step,
        )

    @property
    def steps(self) -> list[int]:
        return self.scheduler.inference_steps

    @abstractmethod
    def set_unet_context(self, *, timestep: Tensor, clip_text_embedding: Tensor, **_: Tensor) -> None:
        ...

    @abstractmethod
    def set_self_attention_guidance(self, enable: bool, scale: float = 1.0) -> None:
        ...

    @abstractmethod
    def has_self_attention_guidance(self) -> bool:
        ...

    @abstractmethod
    def compute_self_attention_guidance(
        self, x: Tensor, noise: Tensor, step: int, *, clip_text_embedding: Tensor, **kwargs: Tensor
    ) -> Tensor:
        ...

    def forward(
        self, x: Tensor, step: int, *, clip_text_embedding: Tensor, condition_scale: float = 7.5, **kwargs: Tensor
    ) -> Tensor:
        timestep = self.scheduler.timesteps[step].unsqueeze(dim=0)
        self.set_unet_context(timestep=timestep, clip_text_embedding=clip_text_embedding, **kwargs)

        latents = torch.cat(tensors=(x, x))  # for classifier-free guidance
        # scale latents for schedulers that need it
        latents = self.scheduler.scale_model_input(latents, step=step)
        unconditional_prediction, conditional_prediction = self.unet(latents).chunk(2)

        # classifier-free guidance
        noise = unconditional_prediction + condition_scale * (conditional_prediction - unconditional_prediction)
        x = x.narrow(dim=1, start=0, length=4)  # support > 4 channels for inpainting

        if self.has_self_attention_guidance():
            noise += self.compute_self_attention_guidance(
                x=x, noise=unconditional_prediction, step=step, clip_text_embedding=clip_text_embedding, **kwargs
            )

        return self.scheduler(x, noise=noise, step=step)

    def structural_copy(self: TLatentDiffusionModel) -> TLatentDiffusionModel:
        return self.__class__(
            unet=self.unet.structural_copy(),
            lda=self.lda.structural_copy(),
            clip_text_encoder=self.clip_text_encoder.structural_copy(),
            scheduler=self.scheduler,
            device=self.device,
            dtype=self.dtype,
        )

# pylama:ignore=W0613
import torch

from imaginairy.img_log import log_latent
from imaginairy.samplers.base import CFGDenoiser
from imaginairy.utils import get_device
from imaginairy.vendored.k_diffusion import sampling as k_sampling
from imaginairy.vendored.k_diffusion.external import CompVisDenoiser


class StandardCompVisDenoiser(CompVisDenoiser):
    def apply_model(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)


class KDiffusionSampler:
    def __init__(self, model, sampler_name):
        self.model = model
        self.cv_denoiser = StandardCompVisDenoiser(model)
        self.sampler_name = sampler_name
        self.sampler_func = getattr(k_sampling, f"sample_{sampler_name}")

    def sample(
        self,
        num_steps,
        conditioning,
        batch_size,
        shape,
        unconditional_guidance_scale,
        unconditional_conditioning,
        eta,
        initial_noise_tensor=None,
        img_callback=None,
    ):
        initial_noise_tensor = (
            torch.randn(shape, device="cpu").to(get_device())
            if initial_noise_tensor is None
            else initial_noise_tensor
        )
        log_latent(initial_noise_tensor, "initial_noise_tensor")

        sigmas = self.cv_denoiser.get_sigmas(num_steps)

        x = initial_noise_tensor * sigmas[0]
        log_latent(x, "initial_sigma_noised_tensor")
        model_wrap_cfg = CFGDenoiser(self.cv_denoiser)

        def callback(data):
            log_latent(data["x"], "x")
            log_latent(data["denoised"], "denoised")

        samples = self.sampler_func(
            model_wrap_cfg,
            x,
            sigmas,
            extra_args={
                "cond": conditioning,
                "uncond": unconditional_conditioning,
                "cond_scale": unconditional_guidance_scale,
            },
            disable=False,
            callback=callback,
        )

        return samples

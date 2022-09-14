import torch
from torch import nn

from imaginairy.img_log import log_latent
from imaginairy.utils import get_device
from imaginairy.vendored.k_diffusion import sampling as k_sampling
from imaginairy.vendored.k_diffusion.external import CompVisDenoiser


class CFGMaskedDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
        x_in = x
        x_in = torch.cat([x_in] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        denoised = uncond + (cond - uncond) * cond_scale

        if mask is not None:
            assert x0 is not None
            img_orig = x0
            mask_inv = 1.0 - mask
            denoised = (img_orig * mask_inv) + (mask * denoised)

        return denoised


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KDiffusionSampler:
    def __init__(self, model, sampler_name):
        self.model = model
        self.cv_denoiser = CompVisDenoiser(model)
        # self.cfg_denoiser = CompVisDenoiser(self.cv_denoiser)
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
        size = (batch_size, *shape)

        initial_noise_tensor = (
            torch.randn(size, device="cpu").to(get_device())
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

        return samples, None

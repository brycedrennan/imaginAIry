import torch
from torch import nn

from imaginairy.samplers.ddim import DDIMSampler
from imaginairy.samplers.kdiff import KDiffusionSampler
from imaginairy.samplers.plms import PLMSSampler
from imaginairy.utils import get_device

SAMPLER_TYPE_OPTIONS = [
    "plms",
    "ddim",
    "k_lms",
    "k_dpm_2",
    "k_dpm_2_a",
    "k_euler",
    "k_euler_a",
    "k_heun",
]

_k_sampler_type_lookup = {
    "k_dpm_2": "dpm_2",
    "k_dpm_2_a": "dpm_2_ancestral",
    "k_euler": "euler",
    "k_euler_a": "euler_ancestral",
    "k_heun": "heun",
    "k_lms": "lms",
}


def get_sampler(sampler_type, model):
    sampler_type = sampler_type.lower()
    if sampler_type == "plms":
        return PLMSSampler(model)
    elif sampler_type == "ddim":
        return DDIMSampler(model)
    elif sampler_type.startswith("k_"):
        sampler_type = _k_sampler_type_lookup[sampler_type]
        return KDiffusionSampler(model, sampler_type)


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


class DiffusionSampler:
    """
    wip

    hope to enforce an api upon samplers
    """

    def __init__(self, noise_prediction_model, sampler_func, device=get_device()):
        self.noise_prediction_model = noise_prediction_model
        self.cfg_noise_prediction_model = CFGDenoiser(noise_prediction_model)
        self.sampler_func = sampler_func
        self.device = device

    def sample(
        self,
        num_steps,
        text_conditioning,
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
        sigmas = self.noise_prediction_model.get_sigmas(num_steps)
        x = initial_noise_tensor * sigmas[0]

        samples = self.sampler_func(
            self.cfg_noise_prediction_model,
            x,
            sigmas,
            extra_args={
                "cond": text_conditioning,
                "uncond": unconditional_conditioning,
                "cond_scale": unconditional_guidance_scale,
            },
            disable=False,
        )

        return samples, None

# pylama:ignore=W0613
import logging
from abc import ABC

import numpy as np
import torch

from imaginairy.log_utils import log_latent
from imaginairy.modules.diffusion.util import (
    extract_into_tensor,
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
)
from imaginairy.utils import get_device

logger = logging.getLogger(__name__)


class SamplerName:
    PLMS = "plms"
    DDIM = "ddim"
    K_DPM_FAST = "k_dpm_fast"
    K_DPM_ADAPTIVE = "k_dpm_adaptive"
    K_LMS = "k_lms"
    K_DPM_2 = "k_dpm_2"
    K_DPM_2_ANCESTRAL = "k_dpm_2_a"
    K_DPMPP_2M = "k_dpmpp_2m"
    K_DPMPP_2S_ANCESTRAL = "k_dpmpp_2s_a"
    K_EULER = "k_euler"
    K_EULER_ANCESTRAL = "k_euler_a"
    K_HEUN = "k_heun"


class ImageSampler(ABC):
    short_name: str
    name: str
    default_steps: int
    default_size: int

    def __init__(self, model):
        self.model = model
        self.device = get_device()


def ensure_4_dim(t: torch.Tensor):
    if len(t.shape) == 3:
        t = t.unsqueeze(dim=0)
    return t


def get_noise_prediction(
    denoise_func,
    noisy_latent,
    time_encoding,
    neutral_conditioning,
    positive_conditioning,
    signal_amplification=7.5,
):
    noisy_latent = ensure_4_dim(noisy_latent)

    noisy_latent_in = torch.cat([noisy_latent] * 2)
    time_encoding_in = torch.cat([time_encoding] * 2)
    if isinstance(positive_conditioning, dict):
        assert isinstance(neutral_conditioning, dict)
        conditioning_in = {}
        for k in positive_conditioning:
            if isinstance(positive_conditioning[k], list):
                conditioning_in[k] = [
                    torch.cat([neutral_conditioning[k][i], positive_conditioning[k][i]])
                    for i in range(len(positive_conditioning[k]))
                ]
            else:
                conditioning_in[k] = torch.cat(
                    [neutral_conditioning[k], positive_conditioning[k]]
                )
    else:
        conditioning_in = torch.cat([neutral_conditioning, positive_conditioning])

    # the k-diffusion samplers actually return the denoised predicted latents but things seem
    # to work anyway
    noise_pred_neutral, noise_pred_positive = denoise_func(
        noisy_latent_in, time_encoding_in, conditioning_in
    ).chunk(2)

    amplified_noise_pred = signal_amplification * (
        noise_pred_positive - noise_pred_neutral
    )
    noise_pred = noise_pred_neutral + amplified_noise_pred

    return noise_pred


def mask_blend(noisy_latent, orig_latent, mask, mask_noise, ts, model):
    """
    Apply a mask to the noisy_latent.

    ts is a decreasing value between 1000 and 1
    """
    assert orig_latent is not None
    log_latent(orig_latent, "orig_latent")
    noised_orig_latent = model.q_sample(orig_latent, ts, mask_noise)

    # this helps prevent the weird disjointed images that can happen with masking
    hint_strength = 1
    # if we're in the first 10% of the steps then don't fully noise the parts
    # of the image we're not changing so that the algorithm can learn from the context
    if ts > 1000:
        hinted_orig_latent = (
            noised_orig_latent * (1 - hint_strength) + orig_latent * hint_strength
        )
        log_latent(hinted_orig_latent, f"hinted_orig_latent {ts}")
    else:
        hinted_orig_latent = noised_orig_latent

    hinted_orig_latent_masked = hinted_orig_latent * mask
    log_latent(hinted_orig_latent_masked, f"hinted_orig_latent_masked {ts}")
    noisy_latent_masked = (1.0 - mask) * noisy_latent
    log_latent(noisy_latent_masked, f"noisy_latent_masked {ts}")
    noisy_latent = hinted_orig_latent_masked + noisy_latent_masked
    log_latent(noisy_latent, f"mask-blended noisy_latent {ts}")
    return noisy_latent


def to_torch(x):
    return x.clone().detach().to(torch.float32).to(get_device())


class NoiseSchedule:
    def __init__(
        self,
        model_num_timesteps,
        model_alphas_cumprod,
        ddim_num_steps,
        ddim_discretize="uniform",
        ddim_eta=0.0,
    ):
        device = get_device()
        if model_alphas_cumprod.shape[0] != model_num_timesteps:
            raise ValueError("alphas have to be defined for each timestep")

        self.alphas_cumprod = to_torch(model_alphas_cumprod)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(model_alphas_cumprod.cpu()))
        self.sqrt_one_minus_alphas_cumprod = to_torch(
            np.sqrt(1.0 - model_alphas_cumprod.cpu())
        )

        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=model_num_timesteps,
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=model_alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
        )
        self.ddim_sigmas = ddim_sigmas.to(torch.float32).to(device)
        self.ddim_alphas = ddim_alphas.to(torch.float32).to(device)
        self.ddim_alphas_prev = ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = (
            np.sqrt(1.0 - ddim_alphas).to(torch.float32).to(device)
        )


@torch.no_grad()
def noise_an_image(init_latent, t, schedule, noise=None):
    # fast, but does not allow for exact reconstruction
    # t serves as an index to gather the correct alphas
    t = t.clamp(0, 1000)
    sqrt_alphas_cumprod = torch.sqrt(schedule.ddim_alphas)
    sqrt_one_minus_alphas_cumprod = schedule.ddim_sqrt_one_minus_alphas

    if noise is None:
        noise = torch.randn_like(init_latent, device="cpu").to(get_device())
    return (
        extract_into_tensor(sqrt_alphas_cumprod, t, init_latent.shape) * init_latent
        + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, init_latent.shape)
        * noise
    )

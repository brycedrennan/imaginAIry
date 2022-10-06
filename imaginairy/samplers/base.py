# pylama:ignore=W0613
import torch
from torch import nn

from imaginairy.img_log import log_latent

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
    from imaginairy.samplers.ddim import DDIMSampler  # noqa
    from imaginairy.samplers.kdiff import KDiffusionSampler  # noqa
    from imaginairy.samplers.plms import PLMSSampler  # noqa

    sampler_type = sampler_type.lower()
    if sampler_type == "plms":
        return PLMSSampler(model)
    if sampler_type == "ddim":
        return DDIMSampler(model)
    if sampler_type.startswith("k_"):
        sampler_type = _k_sampler_type_lookup[sampler_type]
        return KDiffusionSampler(model, sampler_type)
    raise ValueError("invalid sampler_type")


class CFGDenoiser(nn.Module):
    """
    Conditional forward guidance wrapper
    """

    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask=None, orig_latent=None):
        def _wrapper(noisy_latent_in, time_encoding_in, conditioning_in):
            return self.inner_model(
                noisy_latent_in, time_encoding_in, cond=conditioning_in
            )

        denoised = get_noise_prediction(
            denoise_func=_wrapper,
            noisy_latent=x,
            time_encoding=sigma,
            neutral_conditioning=uncond,
            positive_conditioning=cond,
            signal_amplification=cond_scale,
        )

        if mask is not None:
            assert orig_latent is not None
            mask_inv = 1.0 - mask
            denoised = (orig_latent * mask_inv) + (mask * denoised)

        return denoised


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
    conditioning_in = torch.cat([neutral_conditioning, positive_conditioning])

    pred_noise_neutral, pred_noise_positive = denoise_func(
        noisy_latent_in, time_encoding_in, conditioning_in
    ).chunk(2)

    amplified_noise_pred = signal_amplification * (
        pred_noise_positive - pred_noise_neutral
    )
    pred_noise = pred_noise_neutral + amplified_noise_pred

    log_latent(pred_noise_neutral, "neutral noise prediction")
    log_latent(pred_noise_positive, "positive noise prediction")
    log_latent(pred_noise, "noise prediction")

    return pred_noise

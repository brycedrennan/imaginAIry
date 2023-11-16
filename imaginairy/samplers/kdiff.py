# pylama:ignore=W0613
from abc import ABC

import torch
from torch import nn

from imaginairy.log_utils import increment_step, log_latent
from imaginairy.samplers.base import (
    ImageSampler,
    SamplerName,
    get_noise_prediction,
    mask_blend,
)
from imaginairy.utils import get_device
from imaginairy.vendored.k_diffusion import sampling as k_sampling
from imaginairy.vendored.k_diffusion.external import CompVisDenoiser, CompVisVDenoiser


class StandardCompVisDenoiser(CompVisDenoiser):
    def apply_model(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)


class StandardCompVisVDenoiser(CompVisVDenoiser):
    def apply_model(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)


def sample_dpm_adaptive(
    model, x, sigmas, extra_args=None, disable=False, callback=None
):
    sigma_min = sigmas[-2]
    sigma_max = sigmas[0]
    return k_sampling.sample_dpm_adaptive(
        model=model,
        x=x,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        extra_args=extra_args,
        disable=disable,
        callback=callback,
    )


def sample_dpm_fast(model, x, sigmas, extra_args=None, disable=False, callback=None):
    sigma_min = sigmas[-2]
    sigma_max = sigmas[0]
    return k_sampling.sample_dpm_fast(
        model=model,
        x=x,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        n=len(sigmas),
        extra_args=extra_args,
        disable=disable,
        callback=callback,
    )


class KDiffusionSampler(ImageSampler, ABC):
    sampler_func: callable

    def __init__(self, model):
        super().__init__(model)
        denoiseer_cls = (
            StandardCompVisVDenoiser
            if getattr(model, "parameterization", "") == "v"
            else StandardCompVisDenoiser
        )
        self.cv_denoiser = denoiseer_cls(model)

    def sample(
        self,
        num_steps,
        shape,
        neutral_conditioning,
        positive_conditioning,
        guidance_scale,
        batch_size=1,
        mask=None,
        orig_latent=None,
        noise=None,
        t_start=None,
        denoiser_cls=None,
    ):
        # if positive_conditioning.shape[0] != batch_size:
        #     raise ValueError(
        #         f"Got {positive_conditioning.shape[0]} conditionings but batch-size is {batch_size}"
        #     )

        if noise is None:
            noise = torch.randn(shape, device="cpu").to(self.device)

        log_latent(noise, "initial noise")
        if t_start is not None:
            t_start = num_steps - t_start + 1
        sigmas = self.cv_denoiser.get_sigmas(num_steps)[t_start:]

        # see https://github.com/crowsonkb/k-diffusion/issues/43#issuecomment-1305195666
        if self.short_name in (
            SamplerName.K_DPM_2,
            SamplerName.K_DPMPP_2M,
            SamplerName.K_DPM_2_ANCESTRAL,
        ):
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        # if our number of steps is zero, just return the initial latent
        if sigmas.nelement() == 0:
            if orig_latent is not None:
                return orig_latent
            return noise

        # t_start is none if init image strength set to 0
        if orig_latent is not None and t_start is not None:
            noisy_latent = noise * sigmas[0] + orig_latent
        else:
            noisy_latent = noise * sigmas[0]

        x = noisy_latent

        log_latent(x, "initial_sigma_noised_tensor")
        if denoiser_cls is None:
            denoiser_cls = CFGDenoiser
        model_wrap_cfg = denoiser_cls(self.cv_denoiser)

        mask_noise = None
        if mask is not None:
            mask_noise = torch.randn_like(x, device="cpu").to(x.device)

        def callback(data):
            log_latent(data["x"], "noisy_latent")
            log_latent(data["denoised"], "predicted_latent")
            increment_step()

        samples = self.sampler_func(
            model=model_wrap_cfg,
            x=x,
            sigmas=sigmas,
            extra_args={
                "cond": positive_conditioning,
                "uncond": neutral_conditioning,
                "cond_scale": guidance_scale,
                "mask": mask,
                "mask_noise": mask_noise,
                "orig_latent": orig_latent,
            },
            disable=False,
            callback=callback,
        )

        return samples


class DPMFastSampler(KDiffusionSampler):
    short_name = SamplerName.K_DPM_FAST
    name = "Diffusion probabilistic models - fast"
    default_steps = 15
    sampler_func = staticmethod(sample_dpm_fast)


class DPMAdaptiveSampler(KDiffusionSampler):
    short_name = SamplerName.K_DPM_ADAPTIVE
    name = "Diffusion probabilistic models - adaptive"
    default_steps = 40
    sampler_func = staticmethod(sample_dpm_adaptive)


class DPM2Sampler(KDiffusionSampler):
    short_name = SamplerName.K_DPM_2
    name = "Diffusion probabilistic models - 2"
    default_steps = 40
    sampler_func = staticmethod(k_sampling.sample_dpm_2)


class DPM2AncestralSampler(KDiffusionSampler):
    short_name = SamplerName.K_DPM_2_ANCESTRAL
    name = "Diffusion probabilistic models - 2 ancestral"
    default_steps = 40
    sampler_func = staticmethod(k_sampling.sample_dpm_2_ancestral)


class DPMPP2MSampler(KDiffusionSampler):
    short_name = SamplerName.K_DPMPP_2M
    name = "Diffusion probabilistic models - 2m"
    default_steps = 15
    sampler_func = staticmethod(k_sampling.sample_dpmpp_2m)


class DPMPP2SAncestralSampler(KDiffusionSampler):
    short_name = SamplerName.K_DPMPP_2S_ANCESTRAL
    name = "Ancestral sampling with DPM-Solver++(2S) second-order steps."
    default_steps = 15
    sampler_func = staticmethod(k_sampling.sample_dpmpp_2s_ancestral)


class EulerSampler(KDiffusionSampler):
    short_name = SamplerName.K_EULER
    name = "Algorithm 2 (Euler steps) from Karras et al. (2022)"
    default_steps = 40
    sampler_func = staticmethod(k_sampling.sample_euler)


class EulerAncestralSampler(KDiffusionSampler):
    short_name = SamplerName.K_EULER_ANCESTRAL
    name = "Euler ancestral"
    default_steps = 40
    sampler_func = staticmethod(k_sampling.sample_euler_ancestral)


class HeunSampler(KDiffusionSampler):
    short_name = SamplerName.K_HEUN
    name = "Algorithm 2 (Heun steps) from Karras et al. (2022)."
    default_steps = 40
    sampler_func = staticmethod(k_sampling.sample_heun)


class LMSSampler(KDiffusionSampler):
    short_name = SamplerName.K_LMS
    name = "LMS"
    default_steps = 40
    sampler_func = staticmethod(k_sampling.sample_lms)


class CFGDenoiser(nn.Module):
    """
    Conditional forward guidance wrapper.
    """

    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.device = get_device()

    def forward(
        self,
        x,
        sigma,
        uncond,
        cond,
        cond_scale,
        mask=None,
        mask_noise=None,
        orig_latent=None,
    ):
        def _wrapper(noisy_latent_in, time_encoding_in, conditioning_in):
            return self.inner_model(
                noisy_latent_in, time_encoding_in, cond=conditioning_in
            )

        if mask is not None:
            assert orig_latent is not None
            t = self.inner_model.sigma_to_t(sigma, quantize=True)
            big_sigma = max(sigma, 1)
            x = mask_blend(
                noisy_latent=x,
                orig_latent=orig_latent * big_sigma,
                mask=mask,
                mask_noise=mask_noise * big_sigma,
                ts=t,
                model=self.inner_model.inner_model,
            )

        noise_pred = get_noise_prediction(
            denoise_func=_wrapper,
            noisy_latent=x,
            time_encoding=sigma,
            neutral_conditioning=uncond,
            positive_conditioning=cond,
            signal_amplification=cond_scale,
        )

        return noise_pred

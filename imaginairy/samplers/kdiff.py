# pylama:ignore=W0613
import torch

from imaginairy.log_utils import increment_step, log_latent
from imaginairy.samplers.base import CFGDenoiser
from imaginairy.utils import get_device
from imaginairy.vendored.k_diffusion import sampling as k_sampling
from imaginairy.vendored.k_diffusion.external import CompVisDenoiser


class StandardCompVisDenoiser(CompVisDenoiser):
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


class KDiffusionSampler:

    sampler_lookup = {
        "dpm_fast": sample_dpm_fast,
        "dpm_adaptive": sample_dpm_adaptive,
        "dpm_2": k_sampling.sample_dpm_2,
        "dpm_2_ancestral": k_sampling.sample_dpm_2_ancestral,
        "dpmpp_2m": k_sampling.sample_dpmpp_2m,
        "dpmpp_2s_ancestral": k_sampling.sample_dpmpp_2s_ancestral,
        "euler": k_sampling.sample_euler,
        "euler_ancestral": k_sampling.sample_euler_ancestral,
        "heun": k_sampling.sample_heun,
        "lms": k_sampling.sample_lms,
    }

    def __init__(self, model, sampler_name):
        self.model = model
        self.cv_denoiser = StandardCompVisDenoiser(model)
        self.sampler_name = sampler_name
        self.sampler_func = self.sampler_lookup[sampler_name]
        self.device = get_device()

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
        initial_latent=None,
        t_start=None,
    ):
        # if positive_conditioning.shape[0] != batch_size:
        #     raise ValueError(
        #         f"Got {positive_conditioning.shape[0]} conditionings but batch-size is {batch_size}"
        #     )

        if initial_latent is None:
            initial_latent = torch.randn(shape, device="cpu").to(self.device)

        log_latent(initial_latent, "initial_latent")
        if t_start is not None:
            t_start = num_steps - t_start + 1

        sigmas = self.cv_denoiser.get_sigmas(num_steps)[t_start:]

        # if our number of steps is zero, just return the initial latent
        if sigmas.nelement() == 0:
            if orig_latent is not None:
                return orig_latent
            return initial_latent

        x = initial_latent * sigmas[0]
        log_latent(x, "initial_sigma_noised_tensor")
        model_wrap_cfg = CFGDenoiser(self.cv_denoiser)

        mask_noise = None
        if mask is not None:
            mask_noise = torch.randn_like(initial_latent, device="cpu").to(
                initial_latent.device
            )

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

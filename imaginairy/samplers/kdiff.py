# pylama:ignore=W0613
import torch

from imaginairy.log_utils import log_latent
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
        if positive_conditioning.shape[0] != batch_size:
            raise ValueError(
                f"Got {positive_conditioning.shape[0]} conditionings but batch-size is {batch_size}"
            )

        if initial_latent is None:
            initial_latent = torch.randn(shape, device="cpu").to(self.device)

        log_latent(initial_latent, "initial_latent")

        sigmas = self.cv_denoiser.get_sigmas(num_steps)

        x = initial_latent * sigmas[0]
        log_latent(x, "initial_sigma_noised_tensor")
        model_wrap_cfg = CFGDenoiser(self.cv_denoiser)

        def callback(data):
            log_latent(data["x"], "noisy_latent")
            log_latent(data["denoised"], "noise_pred")

        samples = self.sampler_func(
            model=model_wrap_cfg,
            x=x,
            sigmas=sigmas,
            extra_args={
                "cond": positive_conditioning,
                "uncond": neutral_conditioning,
                "cond_scale": guidance_scale,
                "mask": mask,
                "orig_latent": orig_latent,
            },
            disable=False,
            callback=callback,
        )

        return samples

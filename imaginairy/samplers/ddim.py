# pylama:ignore=W0613
import logging

import numpy as np
import torch
from tqdm import tqdm

from imaginairy.log_utils import increment_step, log_latent
from imaginairy.modules.diffusion.util import extract_into_tensor, noise_like
from imaginairy.samplers.base import (
    ImageSampler,
    NoiseSchedule,
    SamplerName,
    get_noise_prediction,
    mask_blend,
)
from imaginairy.utils import get_device

logger = logging.getLogger(__name__)


class DDIMSampler(ImageSampler):
    """
    Denoising Diffusion Implicit Models.

    https://arxiv.org/abs/2010.02502
    """

    short_name = SamplerName.DDIM
    name = "Denoising Diffusion Implicit Models"
    default_steps = 40

    @torch.no_grad()
    def sample(
        self,
        num_steps,
        shape,
        neutral_conditioning,
        positive_conditioning,
        guidance_scale=1.0,
        batch_size=1,
        mask=None,
        orig_latent=None,
        temperature=1.0,
        noise_dropout=0.0,
        noise=None,
        t_start=None,
        quantize_x0=False,
        **kwargs,
    ):
        schedule = NoiseSchedule(
            model_num_timesteps=self.model.num_timesteps,
            model_alphas_cumprod=self.model.alphas_cumprod,
            ddim_num_steps=num_steps,
            ddim_discretize="uniform",
        )

        if noise is None:
            noise = torch.randn(shape, device="cpu").to(self.device)

        log_latent(noise, "initial noise")

        timesteps = schedule.ddim_timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        # t_start is none if init image strength set to 0
        if orig_latent is not None and t_start is not None:
            noisy_latent = self.noise_an_image(
                init_latent=orig_latent, t=t_start - 1, schedule=schedule, noise=noise
            )
        else:
            noisy_latent = noise

        mask_noise = None
        if mask is not None:
            mask_noise = torch.randn_like(noisy_latent, device="cpu").to(
                noisy_latent.device
            )

        for i, step in enumerate(tqdm(time_range, total=total_steps)):
            index = total_steps - i - 1
            ts = torch.full((batch_size,), step, device=self.device, dtype=torch.long)

            if mask is not None:
                noisy_latent = mask_blend(
                    noisy_latent=noisy_latent,
                    orig_latent=orig_latent,
                    mask=mask,
                    mask_noise=mask_noise,
                    ts=ts,
                    model=self.model,
                )

            noisy_latent, predicted_latent = self.p_sample_ddim(
                noisy_latent=noisy_latent,
                neutral_conditioning=neutral_conditioning,
                positive_conditioning=positive_conditioning,
                guidance_scale=guidance_scale,
                time_encoding=ts,
                index=index,
                schedule=schedule,
                quantize_denoised=quantize_x0,
                temperature=temperature,
                noise_dropout=noise_dropout,
            )

            log_latent(noisy_latent, "noisy_latent")
            log_latent(predicted_latent, "predicted_latent")
            increment_step()

        return noisy_latent

    def p_sample_ddim(
        self,
        noisy_latent,
        neutral_conditioning,
        positive_conditioning,
        guidance_scale,
        time_encoding,
        index,
        schedule,
        repeat_noise=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        loss_function=None,
    ):
        assert guidance_scale >= 1
        noise_pred = get_noise_prediction(
            denoise_func=self.model.apply_model,
            noisy_latent=noisy_latent,
            time_encoding=time_encoding,
            neutral_conditioning=neutral_conditioning,
            positive_conditioning=positive_conditioning,
            signal_amplification=guidance_scale,
        )

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(
                noisy_latent, time_encoding, noise_pred
            )
        else:
            e_t = noise_pred

        batch_size = noisy_latent.shape[0]

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full(
            (batch_size, 1, 1, 1),
            schedule.ddim_alphas[index],
            device=noisy_latent.device,
        )
        a_prev = torch.full(
            (batch_size, 1, 1, 1),
            schedule.ddim_alphas_prev[index],
            device=noisy_latent.device,
        )
        sigma_t = torch.full(
            (batch_size, 1, 1, 1),
            schedule.ddim_sigmas[index],
            device=noisy_latent.device,
        )
        sqrt_one_minus_at = torch.full(
            (batch_size, 1, 1, 1),
            schedule.ddim_sqrt_one_minus_alphas[index],
            device=noisy_latent.device,
        )

        noisy_latent, predicted_latent = self._p_sample_ddim_formula(
            model=self.model,
            noisy_latent=noisy_latent,
            noise_pred=noise_pred,
            e_t=e_t,
            sqrt_one_minus_at=sqrt_one_minus_at,
            a_t=a_t,
            time_encoding=time_encoding,
            sigma_t=sigma_t,
            a_prev=a_prev,
            noise_dropout=noise_dropout,
            repeat_noise=repeat_noise,
            temperature=temperature,
        )
        return noisy_latent, predicted_latent

    @staticmethod
    def _p_sample_ddim_formula(
        model,
        noisy_latent,
        noise_pred,
        e_t,
        sqrt_one_minus_at,
        a_t,
        time_encoding,
        sigma_t,
        a_prev,
        noise_dropout,
        repeat_noise,
        temperature,
    ):
        if model.parameterization != "v":
            predicted_latent = (noisy_latent - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            predicted_latent = model.predict_start_from_z_and_v(
                noisy_latent, time_encoding, noise_pred
            )
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = (
            sigma_t
            * noise_like(noisy_latent.shape, noisy_latent.device, repeat_noise)
            * temperature
        )
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * predicted_latent + dir_xt + noise
        return x_prev, predicted_latent

    @torch.no_grad()
    def noise_an_image(self, init_latent, t, schedule, noise=None):
        if isinstance(t, int):
            t = torch.tensor([t], device=get_device())
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

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


class PLMSSampler(ImageSampler):
    """
    probabilistic least-mean-squares.

    Provenance:
    https://github.com/CompVis/latent-diffusion/commit/f0c4e092c156986e125f48c61a0edd38ba8ad059
    https://arxiv.org/abs/2202.09778
    https://github.com/luping-liu/PNDM
    """

    short_name = SamplerName.PLMS
    name = "probabilistic least-mean-squares sampler"
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
        quantize_denoised=False,
        **kwargs,
    ):
        # if positive_conditioning.shape[0] != batch_size:
        #     raise ValueError(
        #         f"Got {positive_conditioning.shape[0]} conditionings but batch-size is {batch_size}"
        #     )

        schedule = NoiseSchedule(
            model_num_timesteps=self.model.num_timesteps,
            ddim_num_steps=num_steps,
            model_alphas_cumprod=self.model.alphas_cumprod,
            ddim_discretize="uniform",
        )

        if noise is None:
            noise = torch.randn(shape, device="cpu").to(self.device)

        log_latent(noise, "initial noise")

        timesteps = schedule.ddim_timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        old_eps = []

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
            ts_next = torch.full(
                (batch_size,),
                time_range[min(i + 1, len(time_range) - 1)],
                device=self.device,
                dtype=torch.long,
            )

            if mask is not None:
                noisy_latent = mask_blend(
                    noisy_latent=noisy_latent,
                    orig_latent=orig_latent,
                    mask=mask,
                    mask_noise=mask_noise,
                    ts=ts,
                    model=self.model,
                )

            noisy_latent, predicted_latent, noise_pred = self.p_sample_plms(
                noisy_latent=noisy_latent,
                neutral_conditioning=neutral_conditioning,
                positive_conditioning=positive_conditioning,
                guidance_scale=guidance_scale,
                time_encoding=ts,
                schedule=schedule,
                index=index,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                old_eps=old_eps,
                t_next=ts_next,
            )
            old_eps.append(noise_pred)
            if len(old_eps) >= 4:
                old_eps.pop(0)

            log_latent(noisy_latent, "noisy_latent")
            log_latent(predicted_latent, "predicted_latent")
            increment_step()

        return noisy_latent

    @torch.no_grad()
    def p_sample_plms(
        self,
        noisy_latent,
        neutral_conditioning,
        positive_conditioning,
        guidance_scale,
        time_encoding,
        schedule: NoiseSchedule,
        index,
        repeat_noise=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        old_eps=None,
        t_next=None,
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
        batch_size = noisy_latent.shape[0]

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            alpha_at_t = torch.full(
                (batch_size, 1, 1, 1), schedule.ddim_alphas[index], device=self.device
            )
            alpha_prev_at_t = torch.full(
                (batch_size, 1, 1, 1),
                schedule.ddim_alphas_prev[index],
                device=self.device,
            )
            sigma_t = torch.full(
                (batch_size, 1, 1, 1), schedule.ddim_sigmas[index], device=self.device
            )
            sqrt_one_minus_at = torch.full(
                (batch_size, 1, 1, 1),
                schedule.ddim_sqrt_one_minus_alphas[index],
                device=self.device,
            )

            # current prediction for x_0
            pred_x0 = (noisy_latent - sqrt_one_minus_at * e_t) / alpha_at_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - alpha_prev_at_t - sigma_t**2).sqrt() * e_t
            noise = (
                sigma_t
                * noise_like(noisy_latent.shape, self.device, repeat_noise)
                * temperature
            )
            if noise_dropout > 0.0:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = alpha_prev_at_t.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(noise_pred, index)
            e_t_next = get_noise_prediction(
                denoise_func=self.model.apply_model,
                noisy_latent=x_prev,
                time_encoding=t_next,
                neutral_conditioning=neutral_conditioning,
                positive_conditioning=positive_conditioning,
                signal_amplification=guidance_scale,
            )
            e_t_prime = (noise_pred + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * noise_pred - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3rd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * noise_pred - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (
                55 * noise_pred - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]
            ) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)
        log_latent(x_prev, "x_prev")
        log_latent(pred_x0, "pred_x0")

        return x_prev, pred_x0, noise_pred

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

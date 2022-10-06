# pylama:ignore=W0613
import logging

import numpy as np
import torch
from tqdm import tqdm

from imaginairy.img_log import log_latent
from imaginairy.modules.diffusion.util import (
    extract_into_tensor,
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)
from imaginairy.samplers.base import get_noise_prediction
from imaginairy.utils import get_device

logger = logging.getLogger(__name__)


class PLMSSchedule:
    def __init__(
        self,
        ddpm_num_timesteps,  # 1000?
        ddim_num_steps,  # prompt.steps?
        alphas_cumprod,
        alphas_cumprod_prev,
        betas,
        ddim_discretize="uniform",
        ddim_eta=0.0,
    ):
        if ddim_eta != 0:
            raise ValueError("ddim_eta must be 0 for PLMS")
        device = get_device()

        assert (
            alphas_cumprod.shape[0] == ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"

        def to_torch(x):
            return x.clone().detach().to(torch.float32).to(device)

        self.betas = to_torch(betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod.cpu()))
        self.sqrt_one_minus_alphas_cumprod = to_torch(
            np.sqrt(1.0 - alphas_cumprod.cpu())
        )
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        self.sqrt_recipm1_alphas_cumprod = to_torch(
            np.sqrt(1.0 / alphas_cumprod.cpu() - 1)
        )

        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=ddpm_num_timesteps,
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
        )
        self.ddim_sigmas = ddim_sigmas.to(torch.float32).to(torch.device(device))
        self.ddim_alphas = ddim_alphas.to(torch.float32).to(torch.device(device))
        self.ddim_alphas_prev = ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = (
            np.sqrt(1.0 - ddim_alphas).to(torch.float32).to(torch.device(device))
        )


class PLMSSampler:
    """probabilistic least-mean-squares"""

    def __init__(self, model):
        self.model = model
        self.device = get_device()

    @torch.no_grad()
    def sample(
        self,
        num_steps,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        orig_latent=None,
        temperature=1.0,
        noise_dropout=0.0,
        initial_latent=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        timesteps=None,
        quantize_denoised=False,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning.shape[0] != batch_size:
            logger.warning(
                f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
            )

        schedule = PLMSSchedule(
            ddpm_num_timesteps=self.model.num_timesteps,
            ddim_num_steps=num_steps,
            alphas_cumprod=self.model.alphas_cumprod,
            alphas_cumprod_prev=self.model.alphas_cumprod_prev,
            betas=self.model.betas,
            ddim_discretize="uniform",
            ddim_eta=0.0,
        )
        device = self.device
        # batch_size = shape[0]
        if initial_latent is None:
            initial_latent = torch.randn(shape, device="cpu").to(device)
        log_latent(initial_latent, "initial latent")
        if timesteps is None:
            timesteps = schedule.ddim_timesteps
        elif timesteps is not None:
            subset_end = (
                int(
                    min(timesteps / schedule.ddim_timesteps.shape[0], 1)
                    * schedule.ddim_timesteps.shape[0]
                )
                - 1
            )
            timesteps = schedule.ddim_timesteps[:subset_end]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        logger.debug(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="    PLMS Sampler", total=total_steps)
        old_eps = []
        img = initial_latent

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            ts_next = torch.full(
                (batch_size,),
                time_range[min(i + 1, len(time_range) - 1)],
                device=device,
                dtype=torch.long,
            )

            if mask is not None:
                assert orig_latent is not None
                img_orig = self.model.q_sample(orig_latent, ts)
                img = img_orig * mask + (1.0 - mask) * img

            img, pred_x0, noise_prediction = self.p_sample_plms(
                img,
                conditioning,
                ts,
                schedule=schedule,
                index=index,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                old_eps=old_eps,
                t_next=ts_next,
            )
            old_eps.append(noise_prediction)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, "img")
                img_callback(pred_x0, "pred_x0")

        return img

    @torch.no_grad()
    def p_sample_plms(
        self,
        noisy_latent,
        positive_conditioning,
        time_encoding,
        schedule: PLMSSchedule,
        index,
        repeat_noise=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
    ):
        batch_size = noisy_latent.shape[0]
        noise_prediction = get_noise_prediction(
            denoise_func=self.model.apply_model,
            noisy_latent=noisy_latent,
            time_encoding=time_encoding,
            neutral_conditioning=unconditional_conditioning,
            positive_conditioning=positive_conditioning,
            signal_amplification=unconditional_guidance_scale,
        )

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
            x_prev, pred_x0 = get_x_prev_and_pred_x0(noise_prediction, index)
            e_t_next = get_noise_prediction(
                denoise_func=self.model.apply_model,
                noisy_latent=x_prev,
                time_encoding=t_next,
                neutral_conditioning=unconditional_conditioning,
                positive_conditioning=positive_conditioning,
                signal_amplification=unconditional_guidance_scale,
            )
            e_t_prime = (noise_prediction + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * noise_prediction - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (
                23 * noise_prediction - 16 * old_eps[-1] + 5 * old_eps[-2]
            ) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (
                55 * noise_prediction
                - 59 * old_eps[-1]
                + 37 * old_eps[-2]
                - 9 * old_eps[-3]
            ) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)
        log_latent(x_prev, "x_prev")
        log_latent(pred_x0, "pred_x0")

        return x_prev, pred_x0, noise_prediction

    @torch.no_grad()
    def noise_an_image(self, init_latent, t, schedule, noise=None):
        # replace with ddpm.q_sample?
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

    @torch.no_grad()
    def decode(
        self,
        cond,
        schedule,
        initial_latent=None,
        t_start=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        img_callback=None,
        temperature=1.0,
        mask=None,
        orig_latent=None,
        noise=None,
    ):
        device = self.device
        timesteps = schedule.ddim_timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc="PLMS img2img", total=total_steps)
        x_dec = initial_latent
        old_eps = []
        log_latent(x_dec, "x_dec")

        # not sure what the downside of using the same noise throughout the process would be...
        # seems to work fine. maybe it runs faster?
        noise = (
            torch.randn_like(x_dec, device="cpu").to(x_dec.device)
            if noise is None
            else noise
        )
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full(
                (initial_latent.shape[0],),
                step,
                device=initial_latent.device,
                dtype=torch.long,
            )
            ts_next = torch.full(
                (initial_latent.shape[0],),
                time_range[min(i + 1, len(time_range) - 1)],
                device=device,
                dtype=torch.long,
            )

            if mask is not None:
                assert orig_latent is not None
                xdec_orig = self.model.q_sample(orig_latent, ts, noise)
                log_latent(xdec_orig, f"xdec_orig i={i} index-{index}")
                # this helps prevent the weird disjointed images that can happen with masking
                hint_strength = 0.8
                if i < 2:
                    xdec_orig_with_hints = (
                        xdec_orig * (1 - hint_strength) + orig_latent * hint_strength
                    )
                else:
                    xdec_orig_with_hints = xdec_orig
                x_dec = xdec_orig_with_hints * mask + (1.0 - mask) * x_dec
                log_latent(x_dec, f"x_dec {ts}")

            x_dec, pred_x0, noise_prediction = self.p_sample_plms(
                x_dec,
                cond,
                ts,
                schedule=schedule,
                index=index,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                temperature=temperature,
                old_eps=old_eps,
                t_next=ts_next,
            )

            old_eps.append(noise_prediction)
            if len(old_eps) >= 4:
                old_eps.pop(0)

            if img_callback:
                img_callback(x_dec, "x_dec")
                img_callback(pred_x0, "pred_x0")

            log_latent(x_dec, f"x_dec {i}")
            log_latent(x_dec, f"e_t {i}")
            log_latent(pred_x0, f"pred_x0 {i}")
        return x_dec

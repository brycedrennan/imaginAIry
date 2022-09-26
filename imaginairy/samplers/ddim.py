# pylama:ignore=W0613
"""SAMPLING ONLY."""
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
from imaginairy.utils import get_device

logger = logging.getLogger(__name__)


class DDIMSampler:
    """
    Denoising Diffusion Implicit Models

    https://arxiv.org/abs/2010.02502
    """

    def __init__(self, model):
        self.model = model

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0):
        buffers = self._make_schedule(
            model_num_timesteps=self.model.num_timesteps,
            model_alphas_cumprod=self.model.alphas_cumprod,
            model_betas=self.model.betas,
            model_alphas_cumprod_prev=self.model.alphas_cumprod_prev,
            ddim_num_steps=ddim_num_steps,
            ddim_discretize=ddim_discretize,
            ddim_eta=ddim_eta,
            device=self.model.device,
        )
        for k, v in buffers.items():
            setattr(self, k, v)

    @staticmethod
    def _make_schedule(
        model_num_timesteps,
        model_alphas_cumprod,
        model_betas,
        model_alphas_cumprod_prev,
        ddim_num_steps,
        ddim_discretize="uniform",
        ddim_eta=0.0,
        device=get_device(),
    ):
        ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=model_num_timesteps,
        )
        alphas_cumprod = model_alphas_cumprod
        if not alphas_cumprod.shape[0] == model_num_timesteps:
            raise ValueError("alphas have to be defined for each timestep")

        def to_torch(x):
            return x.clone().detach().to(torch.float32).to(device)

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=ddim_timesteps,
            eta=ddim_eta,
        )

        buffers = {
            "ddim_timesteps": ddim_timesteps,
            "betas": to_torch(model_betas),
            "alphas_cumprod": to_torch(alphas_cumprod),
            "alphas_cumprod_prev": to_torch(model_alphas_cumprod_prev),
            # calculations for diffusion q(x_t | x_{t-1}) and others
            "sqrt_alphas_cumprod": to_torch(np.sqrt(alphas_cumprod.cpu())),
            "sqrt_one_minus_alphas_cumprod": to_torch(
                np.sqrt(1.0 - alphas_cumprod.cpu())
            ),
            "log_one_minus_alphas_cumprod": to_torch(
                np.log(1.0 - alphas_cumprod.cpu())
            ),
            "sqrt_recip_alphas_cumprod": to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())),
            "sqrt_recipm1_alphas_cumprod": to_torch(
                np.sqrt(1.0 / alphas_cumprod.cpu() - 1)
            ),
            "ddim_sigmas": ddim_sigmas.to(torch.float32).to(device),
            "ddim_alphas": ddim_alphas.to(torch.float32).to(device),
            "ddim_alphas_prev": ddim_alphas_prev,
            "ddim_sqrt_one_minus_alphas": np.sqrt(1.0 - ddim_alphas)
            .to(torch.float32)
            .to(device),
        }

        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - buffers["alphas_cumprod_prev"])
            / (1 - buffers["alphas_cumprod"])
            * (1 - buffers["alphas_cumprod"] / buffers["alphas_cumprod_prev"])
        )
        buffers[
            "ddim_sigmas_for_original_num_steps"
        ] = sigmas_for_original_sampling_steps
        return buffers

    @torch.no_grad()
    def sample(
        self,
        num_steps,
        batch_size,
        shape,
        conditioning,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        x_T=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **kwargs,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
    ):
        if isinstance(conditioning, dict):
            cbs = conditioning[list(conditioning.keys())[0]].shape[0]
            if cbs != batch_size:
                logger.warning(
                    f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                )
        else:
            if conditioning.shape[0] != batch_size:
                logger.warning(
                    f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                )

        self.make_schedule(ddim_num_steps=num_steps, ddim_eta=eta)

        samples = self.ddim_sampling(
            conditioning,
            shape=(batch_size, *shape),
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )
        return samples

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            # run on CPU for seed consistency. M1/mps runs were not consistent otherwise
            img = torch.randn(shape, device="cpu").to(device)
        else:
            img = x_T
        log_latent(img, "initial noise")

        if timesteps is None:
            timesteps = self.ddim_timesteps
        else:
            subset_end = (
                int(
                    min(timesteps / self.ddim_timesteps.shape[0], 1)
                    * self.ddim_timesteps.shape[0]
                )
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        logger.info(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(
                    x0, ts
                )  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            img, pred_x0 = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            if callback:
                callback(i)

            log_latent(img, "img")
            log_latent(pred_x0, "pred_x0")

        return img

    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        loss_function=None,
    ):
        assert unconditional_guidance_scale >= 1
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([unconditional_conditioning, c])
        # with torch.no_grad():
        noise_pred_uncond, noise_pred = self.model.apply_model(x_in, t_in, c_in).chunk(
            2
        )
        noise_pred = noise_pred_uncond + unconditional_guidance_scale * (
            noise_pred - noise_pred_uncond
        )

        b = x.shape[0]
        log_latent(noise_pred, "noise prediction")

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=x.device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=x.device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=x.device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=x.device
        )
        return self._p_sample_ddim_formula(
            x,
            noise_pred,
            sqrt_one_minus_at,
            a_t,
            sigma_t,
            a_prev,
            noise_dropout,
            repeat_noise,
            temperature,
        )

    @staticmethod
    def _p_sample_ddim_formula(
        x,
        noise_pred,
        sqrt_one_minus_at,
        a_t,
        sigma_t,
        a_prev,
        noise_dropout,
        repeat_noise,
        temperature,
    ):
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * noise_pred
        noise = sigma_t * noise_like(x.shape, x.device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, init_latent, t, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        t = t.clamp(0, 1000)
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

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
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        img_callback=None,
        score_corrector=None,
        temperature=1.0,
        mask=None,
        orig_latent=None,
    ):

        timesteps = self.ddim_timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        logger.debug(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full(
                (x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long
            )

            if mask is not None:
                assert orig_latent is not None
                xdec_orig = self.model.q_sample(orig_latent, ts)
                log_latent(xdec_orig, "xdec_orig")
                # this helps prevent the weird disjointed images that can happen with masking
                hint_strength = 0.8
                xdec_orig_with_hints = (
                    xdec_orig * (1 - hint_strength) + orig_latent * hint_strength
                )
                x_dec = xdec_orig_with_hints * mask + (1.0 - mask) * x_dec
                log_latent(x_dec, "x_dec")

            x_dec, pred_x0 = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                temperature=temperature,
            )
            # original_loss = ((x_dec - x_latent).abs().mean()*70)
            # sigma_t = torch.full((1, 1, 1, 1), self.ddim_sigmas[index], device=get_device())
            # x_dec = x_dec.detach() + (original_loss * 0.1) ** 2
            # cond_grad = -torch.autograd.grad(original_loss, x_dec)[0]
            # x_dec = x_dec.detach() + cond_grad * sigma_t ** 2
            # x_dec_alt = x_dec + (original_loss * 0.1) ** 2

            log_latent(x_dec, f"x_dec {i}")
            log_latent(pred_x0, f"pred_x0 {i}")
        return x_dec

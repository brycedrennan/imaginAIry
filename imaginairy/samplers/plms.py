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


class PLMSSampler:
    """probabilistic least-mean-squares"""

    def __init__(self, model):
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.device_available = get_device()
        self.ddim_timesteps = None

    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            if attr.device != torch.device(self.device_available):
                attr = attr.to(torch.float32).to(torch.device(self.device_available))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0):
        if ddim_eta != 0:
            raise ValueError("ddim_eta must be 0 for PLMS")
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"

        def to_torch(x):
            return x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    @torch.no_grad()
    def sample(
        self,
        num_steps,
        batch_size,
        shape,
        conditioning=None,
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
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
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

        samples = self.plms_sampling(
            conditioning,
            (batch_size, *shape),
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
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
    def plms_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
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

            img = torch.randn(shape, device="cpu").to(device)
        else:
            img = x_T
        log_latent(img, "initial img")
        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (
                int(
                    min(timesteps / self.ddim_timesteps.shape[0], 1)
                    * self.ddim_timesteps.shape[0]
                )
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        time_range = (
            list(reversed(range(0, timesteps)))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        logger.debug(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="    PLMS Sampler", total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full(
                (b,),
                time_range[min(i + 1, len(time_range) - 1)],
                device=device,
                dtype=torch.long,
            )

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(
                    x0, ts
                )  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            img, pred_x0, e_t = self.p_sample_plms(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                old_eps=old_eps,
                t_next=ts_next,
            )
            old_eps.append(e_t)
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
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
    ):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if (
                unconditional_conditioning is None
                or unconditional_guidance_scale == 1.0
            ):
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                log_latent(e_t_uncond, "noise pred uncond")
                log_latent(e_t, "noise pred cond")

                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                log_latent(e_t, "noise pred combined")

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(
                    self.model, e_t, x, t, c, **corrector_kwargs
                )

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.0:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (
                55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]
            ) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)
        log_latent(x_prev, "x_prev")
        log_latent(pred_x0, "pred_x0")

        return x_prev, pred_x0, e_t

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
        noise=None,
    ):

        timesteps = self.ddim_timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc="PLMS altering image", total=total_steps)
        x_dec = x_latent
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
                (x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long
            )
            ts_next = torch.full(
                (x_latent.shape[0],),
                time_range[min(i + 1, len(time_range) - 1)],
                device=x_latent.device,
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

            x_dec, pred_x0, e_t = self.p_sample_plms(
                x_dec,
                cond,
                ts,
                index=index,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                temperature=temperature,
                old_eps=old_eps,
                t_next=ts_next,
            )
            # original_loss = ((x_dec - x_latent).abs().mean()*70)
            # sigma_t = torch.full((1, 1, 1, 1), self.ddim_sigmas[index], device=get_device())
            # x_dec = x_dec.detach() + (original_loss * 0.1) ** 2
            # cond_grad = -torch.autograd.grad(original_loss, x_dec)[0]
            # x_dec = x_dec.detach() + cond_grad * sigma_t ** 2
            # x_dec_alt = x_dec + (original_loss * 0.1) ** 2

            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)

            if img_callback:
                img_callback(x_dec, "x_dec")
                img_callback(pred_x0, "pred_x0")

            log_latent(x_dec, f"x_dec {i}")
            log_latent(x_dec, f"e_t {i}")
            log_latent(pred_x0, f"pred_x0 {i}")
        return x_dec

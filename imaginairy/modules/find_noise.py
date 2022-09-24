"""
I tried it with the DDIM sampler and it didn't work.

Probably need to use the k-diffusion sampler with it
from https://gist.githubusercontent.com/trygvebw/c71334dd127d537a15e9d59790f7f5e1/raw/a846393251f5be8289d4febc75a19f1f962aabcc/find_noise.py

needs https://github.com/crowsonkb/k-diffusion
"""
from contextlib import nullcontext

import torch
from torch import autocast

from imaginairy.img_utils import pillow_img_to_model_latent
from imaginairy.utils import get_device
from imaginairy.vendored import k_diffusion as K


def find_noise_for_image(model, pil_img, prompt, steps=50, cond_scale=1.0, half=True):
    img_latent = pillow_img_to_model_latent(model, pil_img, batch_size=1, half=half)
    return find_noise_for_latent(
        model,
        img_latent,
        prompt,
        steps=steps,
        cond_scale=cond_scale,
    )


def find_noise_for_latent(model, img_latent, prompt, steps=50, cond_scale=1.0):
    x = img_latent

    _autocast = autocast if get_device() in ("cuda", "cpu") else nullcontext
    with (torch.no_grad(), _autocast(get_device())):
        uncond = model.get_learned_conditioning([""])
        cond = model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    with (torch.no_grad(), _autocast(get_device())):
        for i in range(1, len(sigmas)):
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigmas[i] * s_in] * 2)
            cond_in = torch.cat([uncond, cond])

            c_out, c_in = [
                K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)
            ]
            t = dnw.sigma_to_t(sigma_in)

            eps = model.apply_model(x_in * c_in, t, cond=cond_in)
            denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

            denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

            d = (x - denoised) / sigmas[i]
            dt = sigmas[i] - sigmas[i - 1]

            x = x + d * dt

            # This shouldn't be necessary, but solved some VRAM issues
            del (
                x_in,
                sigma_in,
                cond_in,
                c_out,
                c_in,
                t,
            )
            del eps, denoised_uncond, denoised_cond, denoised, d, dt
            # collect_and_empty()

        # return (x / x.std())
        return (x / x.std()) * sigmas[-1]


if __name__ == "__main__":
    pass

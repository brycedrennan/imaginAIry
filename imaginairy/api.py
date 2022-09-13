import logging
import os
import re
from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import torch
import torch.nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFilter
from pytorch_lightning import seed_everything
from torch import autocast
from transformers import cached_path

from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
from imaginairy.enhancers.upscale_realesrgan import upscale_image
from imaginairy.modules.diffusion.ddim import DDIMSampler
from imaginairy.modules.diffusion.plms import PLMSSampler
from imaginairy.safety import is_nsfw, safety_models
from imaginairy.schema import ImaginePrompt, ImagineResult
from imaginairy.utils import (
    fix_torch_nn_layer_norm,
    get_device,
    img_path_to_torch_image,
    instantiate_from_config,
)

LIB_PATH = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

# leave undocumented. I'd ask that no one publicize this flag
IMAGINAIRY_ALLOW_NSFW = os.getenv("IMAGINAIRY_ALLOW_NSFW", "False")
IMAGINAIRY_ALLOW_NSFW = bool(IMAGINAIRY_ALLOW_NSFW == "I AM A RESPONSIBLE ADULT")


def load_model_from_config(config):
    url = "https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media"
    ckpt_path = cached_path(url)
    logger.info(f"Loading model onto {get_device()} backend...")
    logger.debug(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        logger.debug(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0:
        logger.debug(f"missing keys: {m}")
    if len(u) > 0:
        logger.debug(f"unexpected keys: {u}")

    model.to(get_device())
    model.eval()
    return model


def patch_conv(**patch):
    """https://github.com/replicate/cog-stable-diffusion/compare/main...TomMoore515:material_stable_diffusion:main"""
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)

    cls.__init__ = __init__


@lru_cache()
def load_model(tile_mode=False):
    if tile_mode:
        # generated images are tileable
        patch_conv(padding_mode="circular")

    config = "configs/stable-diffusion-v1.yaml"
    config = OmegaConf.load(f"{LIB_PATH}/{config}")
    model = load_model_from_config(config)

    model = model.to(get_device())
    return model


def imagine_image_files(
    prompts,
    outdir,
    latent_channels=4,
    downsampling_factor=8,
    precision="autocast",
    ddim_eta=0.0,
    record_step_images=False,
    output_file_extension="jpg",
    tile_mode=False,
):
    big_path = os.path.join(outdir, "upscaled")
    os.makedirs(outdir, exist_ok=True)

    base_count = len(os.listdir(outdir))
    step_count = 0
    output_file_extension = output_file_extension.lower()
    if output_file_extension not in {"jpg", "png"}:
        raise ValueError("Must output a png or jpg")

    def _record_steps(samples, description, model, prompt):
        nonlocal step_count
        step_count += 1
        samples = model.decode_first_stage(samples)
        samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
        steps_path = os.path.join(outdir, "steps", f"{base_count:08}_S{prompt.seed}")
        os.makedirs(steps_path, exist_ok=True)
        for pred_x0 in samples:
            pred_x0 = 255.0 * rearrange(pred_x0.cpu().numpy(), "c h w -> h w c")
            filename = f"{base_count:08}_S{prompt.seed}_step{step_count:04}.jpg"
            img = Image.fromarray(pred_x0.astype(np.uint8))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), str(description))
            img.save(os.path.join(steps_path, filename))

    img_callback = _record_steps if record_step_images else None
    for result in imagine(
        prompts,
        latent_channels=latent_channels,
        downsampling_factor=downsampling_factor,
        precision=precision,
        ddim_eta=ddim_eta,
        img_callback=img_callback,
        tile_mode=tile_mode,
    ):
        prompt = result.prompt
        basefilename = f"{base_count:06}_{prompt.seed}_{prompt.sampler_type}{prompt.steps}_PS{prompt.prompt_strength}_{prompt_normalized(prompt.prompt_text)}"
        filepath = os.path.join(outdir, f"{basefilename}.jpg")

        result.save(filepath)
        logger.info(f"    🖼  saved to: {filepath}")
        if result.upscaled_img:
            os.makedirs(big_path, exist_ok=True)
            bigfilepath = os.path.join(big_path, basefilename) + "_upscaled.jpg"
            result.save_upscaled(bigfilepath)
            logger.info(f"    Upscaled 🖼  saved to: {bigfilepath}")
        base_count += 1


def imagine(
    prompts,
    latent_channels=4,
    downsampling_factor=8,
    precision="autocast",
    ddim_eta=0.0,
    img_callback=None,
    tile_mode=False,
    half_mode=None,
):
    model = load_model(tile_mode=tile_mode)

    if not IMAGINAIRY_ALLOW_NSFW:
        # needs to be loaded before we set default tensor type to half
        safety_models()
    # only run half-mode on cuda. run it by default
    half_mode = half_mode is None and get_device() == "cuda"
    if half_mode:
        model = model.half()
        # needed when model is in half mode, remove if not using half mode
        # torch.set_default_tensor_type(torch.HalfTensor)
    prompts = [ImaginePrompt(prompts)] if isinstance(prompts, str) else prompts
    prompts = [prompts] if isinstance(prompts, ImaginePrompt) else prompts
    _img_callback = None

    precision_scope = (
        autocast
        if precision == "autocast" and get_device() in ("cuda", "cpu")
        else nullcontext
    )
    with (torch.no_grad(), precision_scope(get_device()), fix_torch_nn_layer_norm()):
        for prompt in prompts:
            logger.info(f"Generating {prompt.prompt_description()}")
            seed_everything(prompt.seed)

            uc = None
            if prompt.prompt_strength != 1.0:
                uc = model.get_learned_conditioning(1 * [""])
            total_weight = sum(wp.weight for wp in prompt.prompts)
            c = sum(
                [
                    model.get_learned_conditioning(wp.text) * (wp.weight / total_weight)
                    for wp in prompt.prompts
                ]
            )

            def _img_callback(samples, description):
                if img_callback:
                    img_callback(samples, description, model, prompt)

            shape = [
                latent_channels,
                prompt.height // downsampling_factor,
                prompt.width // downsampling_factor,
            ]

            start_code = None
            sampler = get_sampler(prompt.sampler_type, model)
            if prompt.init_image:
                generation_strength = 1 - prompt.init_image_strength
                ddim_steps = int(prompt.steps / generation_strength)
                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta)

                t_enc = int(generation_strength * ddim_steps)
                init_image, w, h = img_path_to_torch_image(prompt.init_image)
                init_image = init_image.to(get_device())
                init_latent = model.get_first_stage_encoding(
                    model.encode_first_stage(init_image)
                )
                _img_callback(init_latent, "init_latent")

                # encode (scaled latent)
                z_enc = sampler.stochastic_encode(
                    init_latent, torch.tensor([t_enc]).to(get_device())
                )
                _img_callback(z_enc, "z_enc")

                # decode it
                samples = sampler.decode(
                    z_enc,
                    c,
                    t_enc,
                    unconditional_guidance_scale=prompt.prompt_strength,
                    unconditional_conditioning=uc,
                    img_callback=_img_callback,
                )
            else:

                samples, _ = sampler.sample(
                    S=prompt.steps,
                    conditioning=c,
                    batch_size=1,
                    shape=shape,
                    unconditional_guidance_scale=prompt.prompt_strength,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    x_T=start_code,
                    img_callback=_img_callback,
                )

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            for x_sample in x_samples:
                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                x_sample_8_orig = x_sample.astype(np.uint8)
                img = Image.fromarray(x_sample_8_orig)
                upscaled_img = None
                if not IMAGINAIRY_ALLOW_NSFW and is_nsfw(
                    img, x_sample, half_mode=half_mode
                ):
                    logger.info("    ⚠️  Filtering NSFW image")
                    img = img.filter(ImageFilter.GaussianBlur(radius=40))

                if prompt.fix_faces:
                    logger.info("    Fixing 😊 's in 🖼  using GFPGAN...")
                    img = enhance_faces(img, fidelity=0.2)
                if prompt.upscale:
                    logger.info("    Upscaling 🖼  using real-ESRGAN...")
                    upscaled_img = upscale_image(img)

                yield ImagineResult(img=img, prompt=prompt, upscaled_img=upscaled_img)


def prompt_normalized(prompt):
    return re.sub(r"[^a-zA-Z0-9.,]+", "_", prompt)[:130]


DOWNLOADED_FILES_PATH = f"{LIB_PATH}/../downloads/"


def get_sampler(sampler_type, model):
    sampler_type = sampler_type.upper()
    if sampler_type == "PLMS":
        return PLMSSampler(model)
    elif sampler_type == "DDIM":
        return DDIMSampler(model)

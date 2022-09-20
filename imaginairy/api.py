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
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pytorch_lightning import seed_everything
from torch import autocast
from transformers import cached_path

from imaginairy.enhancers.clip_masking import get_img_mask
from imaginairy.enhancers.describe_image_blip import generate_caption
from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
from imaginairy.enhancers.upscale_realesrgan import upscale_image
from imaginairy.img_log import (
    ImageLoggingContext,
    log_conditioning,
    log_img,
    log_latent,
)
from imaginairy.safety import is_nsfw
from imaginairy.samplers.base import get_sampler
from imaginairy.schema import ImaginePrompt, ImagineResult
from imaginairy.utils import (
    expand_mask,
    fix_torch_nn_layer_norm,
    get_device,
    instantiate_from_config,
    pillow_fit_image_within,
    pillow_img_to_torch_image,
)

LIB_PATH = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


class SafetyMode:
    DISABLED = "disabled"
    CLASSIFY = "classify"
    FILTER = "filter"


# leave undocumented. I'd ask that no one publicize this flag. Just want a
# slight barrier to entry. Please don't use this is any way that's gonna cause
# the press or governments to freak out about AI...
IMAGINAIRY_SAFETY_MODE = os.getenv("IMAGINAIRY_SAFETY_MODE", SafetyMode.FILTER)


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
    """
    Patch to enable tiling mode

    https://github.com/replicate/cog-stable-diffusion/compare/main...TomMoore515:material_stable_diffusion:main
    """
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
    print_caption=False,
):
    big_path = os.path.join(outdir, "upscaled")
    os.makedirs(outdir, exist_ok=True)

    base_count = len(os.listdir(outdir))
    output_file_extension = output_file_extension.lower()
    if output_file_extension not in {"jpg", "png"}:
        raise ValueError("Must output a png or jpg")

    def _record_step(img, description, step_count, prompt):
        steps_path = os.path.join(outdir, "steps", f"{base_count:08}_S{prompt.seed}")
        os.makedirs(steps_path, exist_ok=True)
        filename = f"{base_count:08}_S{prompt.seed}_step{step_count:04}_{prompt_normalized(description)[:40]}.jpg"

        destination = os.path.join(steps_path, filename)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), str(description))
        img.save(destination)

    for result in imagine(
        prompts,
        latent_channels=latent_channels,
        downsampling_factor=downsampling_factor,
        precision=precision,
        ddim_eta=ddim_eta,
        img_callback=_record_step if record_step_images else None,
        tile_mode=tile_mode,
        add_caption=print_caption,
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
        del result


def imagine(
    prompts,
    latent_channels=4,
    downsampling_factor=8,
    precision="autocast",
    ddim_eta=0.0,
    img_callback=None,
    tile_mode=False,
    half_mode=None,
    add_caption=False,
):
    model = load_model(tile_mode=tile_mode)

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
    with torch.no_grad(), precision_scope(get_device()), fix_torch_nn_layer_norm():
        for prompt in prompts:
            with ImageLoggingContext(
                prompt=prompt,
                model=model,
                img_callback=img_callback,
            ):
                logger.info(f"Generating {prompt.prompt_description()}")
                seed_everything(prompt.seed)

                uc = None
                if prompt.prompt_strength != 1.0:
                    uc = model.get_learned_conditioning(1 * [""])
                    log_conditioning(uc, "neutral conditioning")
                if prompt.conditioning is not None:
                    c = prompt.conditioning
                else:
                    total_weight = sum(wp.weight for wp in prompt.prompts)
                    c = sum(
                        model.get_learned_conditioning(wp.text)
                        * (wp.weight / total_weight)
                        for wp in prompt.prompts
                    )
                log_conditioning(c, "positive conditioning")

                shape = [
                    latent_channels,
                    prompt.height // downsampling_factor,
                    prompt.width // downsampling_factor,
                ]
                if prompt.init_image and prompt.sampler_type not in ("ddim", "plms"):
                    sampler_type = "plms"
                    logger.info("   Sampler type switched to plms for img2img")
                else:
                    sampler_type = prompt.sampler_type
                start_code = None
                sampler = get_sampler(sampler_type, model)
                mask, mask_image, mask_image_orig = None, None, None
                if prompt.init_image:
                    generation_strength = 1 - prompt.init_image_strength
                    ddim_steps = int(prompt.steps / generation_strength)
                    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta)
                    init_image, _, h = pillow_fit_image_within(
                        prompt.init_image,
                        max_height=prompt.height,
                        max_width=prompt.width,
                    )

                    init_image_t = pillow_img_to_torch_image(init_image)

                    if prompt.mask_prompt:
                        mask_image = get_img_mask(init_image, prompt.mask_prompt)
                    elif prompt.mask_image:
                        mask_image = prompt.mask_image

                    if mask_image is not None:
                        log_img(mask_image, "init mask")
                        mask_image = expand_mask(mask_image, prompt.mask_expansion)
                        log_img(mask_image, "init mask expanded")
                        if prompt.mask_mode == ImaginePrompt.MaskMode.REPLACE:
                            mask_image = ImageOps.invert(mask_image)

                        log_img(
                            Image.composite(init_image, mask_image, mask_image),
                            "mask overlay",
                        )
                        mask_image_orig = mask_image
                        mask_image = mask_image.resize(
                            (
                                mask_image.width // downsampling_factor,
                                mask_image.height // downsampling_factor,
                            ),
                            resample=Image.Resampling.NEAREST,
                        )
                        log_img(mask_image, "init mask 2")

                        mask = np.array(mask_image)
                        mask = mask.astype(np.float32) / 255.0
                        mask = mask[None, None]
                        mask[mask < 0.9] = 0
                        mask[mask >= 0.9] = 1
                        mask = torch.from_numpy(mask)
                        mask = mask.to(get_device())

                    init_image_t = init_image_t.to(get_device())
                    init_latent = model.get_first_stage_encoding(
                        model.encode_first_stage(init_image_t)
                    )

                    log_latent(init_latent, "init_latent")
                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(
                        init_latent,
                        torch.tensor([prompt.steps]).to(get_device()),
                    )
                    log_latent(z_enc, "z_enc")

                    # decode it
                    samples = sampler.decode(
                        z_enc,
                        c,
                        prompt.steps,
                        unconditional_guidance_scale=prompt.prompt_strength,
                        unconditional_conditioning=uc,
                        img_callback=_img_callback,
                        mask=mask,
                        orig_latent=init_latent,
                    )
                else:

                    samples = sampler.sample(
                        num_steps=prompt.steps,
                        conditioning=c,
                        batch_size=1,
                        shape=shape,
                        unconditional_guidance_scale=prompt.prompt_strength,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        initial_noise_tensor=start_code,
                        img_callback=_img_callback,
                    )

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    x_sample_8_orig = x_sample.astype(np.uint8)
                    img = Image.fromarray(x_sample_8_orig)
                    if mask_image_orig and init_image:

                        mask_image_orig = expand_mask(mask_image_orig, -3)
                        mask_image_orig = mask_image_orig.filter(
                            ImageFilter.GaussianBlur(radius=3)
                        )
                        log_img(mask_image_orig, "reconstituting mask")
                        mask_image_orig = ImageOps.invert(mask_image_orig)
                        img = Image.composite(img, init_image, mask_image_orig)
                        log_img(img, "reconstituted image")

                    upscaled_img = None
                    is_nsfw_img = None
                    if add_caption:
                        caption = generate_caption(img)
                        logger.info(f"    Generated caption: {caption}")
                    if IMAGINAIRY_SAFETY_MODE != SafetyMode.DISABLED:
                        is_nsfw_img = is_nsfw(img, x_sample)
                        if is_nsfw_img and IMAGINAIRY_SAFETY_MODE == SafetyMode.FILTER:
                            logger.info("    ⚠️  Filtering NSFW image")
                            img = img.filter(ImageFilter.GaussianBlur(radius=40))

                    if prompt.fix_faces:
                        logger.info("    Fixing 😊 's in 🖼  using CodeFormer...")
                        img = enhance_faces(img, fidelity=0.2)
                    if prompt.upscale:
                        logger.info("    Upscaling 🖼  using real-ESRGAN...")
                        upscaled_img = upscale_image(img)
                        if prompt.fix_faces:
                            logger.info("    Fixing 😊 's in big 🖼  using CodeFormer...")
                            upscaled_img = enhance_faces(upscaled_img, fidelity=0.8)

                    yield ImagineResult(
                        img=img,
                        prompt=prompt,
                        upscaled_img=upscaled_img,
                        is_nsfw=is_nsfw_img,
                    )


def prompt_normalized(prompt):
    return re.sub(r"[^a-zA-Z0-9.,]+", "_", prompt)[:130]

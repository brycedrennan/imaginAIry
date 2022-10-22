import logging
import os
import re
from functools import lru_cache

import numpy as np
import PIL
import torch
import torch.nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pytorch_lightning import seed_everything
from transformers import cached_path

from imaginairy.enhancers.clip_masking import get_img_mask
from imaginairy.enhancers.describe_image_blip import generate_caption
from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
from imaginairy.enhancers.upscale_realesrgan import upscale_image
from imaginairy.img_utils import pillow_fit_image_within, pillow_img_to_torch_image
from imaginairy.log_utils import (
    ImageLoggingContext,
    log_conditioning,
    log_img,
    log_latent,
)
from imaginairy.safety import SafetyMode, create_safety_score
from imaginairy.samplers.base import NoiseSchedule, get_sampler, noise_an_image
from imaginairy.schema import ImaginePrompt, ImagineResult
from imaginairy.utils import (
    fix_torch_group_norm,
    fix_torch_nn_layer_norm,
    get_device,
    instantiate_from_config,
    platform_appropriate_autocast,
    randn_seeded,
)

LIB_PATH = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

# leave undocumented. I'd ask that no one publicize this flag. Just want a
# slight barrier to entry. Please don't use this is any way that's gonna cause
# the media or politicians to freak out about AI...
IMAGINAIRY_SAFETY_MODE = os.getenv("IMAGINAIRY_SAFETY_MODE", SafetyMode.STRICT)
if IMAGINAIRY_SAFETY_MODE in {"disabled", "classify"}:
    IMAGINAIRY_SAFETY_MODE = SafetyMode.RELAXED
elif IMAGINAIRY_SAFETY_MODE == "filter":
    IMAGINAIRY_SAFETY_MODE = SafetyMode.STRICT

MODEL_LOCATIONS = {
    "sd-1.4": "https://huggingface.co/bstddev/sd-v1-4/resolve/main/sd-v1-4.ckpt",
    "sd-1.5": "https://huggingface.co/acheong08/SD-V1-5-cloned/resolve/main/v1-5-pruned-emaonly.ckpt",
}

DEFAULT_MODEL = "sd-1.5"
DEFAULT_MODEL_WEIGHTS_LOCATION = MODEL_LOCATIONS[DEFAULT_MODEL]


def load_model_from_config(
    config, model_weights_location=DEFAULT_MODEL_WEIGHTS_LOCATION
):
    model_weights_location = (
        model_weights_location
        if model_weights_location
        else DEFAULT_MODEL_WEIGHTS_LOCATION
    )
    if model_weights_location.startswith("http"):
        ckpt_path = cached_path(model_weights_location)
    else:
        ckpt_path = model_weights_location
    logger.info(f"Loading model {ckpt_path} onto {get_device()} backend...")
    pl_sd = None
    try:
        pl_sd = torch.load(ckpt_path, map_location="cpu")
    except RuntimeError as e:
        if "PytorchStreamReader failed reading zip archive" in str(e):
            if model_weights_location.startswith("http"):
                logger.warning("Corrupt checkpoint. deleting and re-downloading...")
                os.remove(ckpt_path)
                ckpt_path = cached_path(model_weights_location)
                pl_sd = torch.load(ckpt_path, map_location="cpu")
        if pl_sd is None:
            raise e
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


@lru_cache()
def load_model(model_weights_location=None):
    config = "configs/stable-diffusion-v1.yaml"
    config = OmegaConf.load(f"{LIB_PATH}/{config}")
    model = load_model_from_config(
        config, model_weights_location=model_weights_location
    )
    model = model.to(get_device())
    return model


def imagine_image_files(
    prompts,
    outdir,
    latent_channels=4,
    downsampling_factor=8,
    precision="autocast",
    record_step_images=False,
    output_file_extension="jpg",
    print_caption=False,
    model_weights_path=None,
):
    generated_imgs_path = os.path.join(outdir, "generated")
    os.makedirs(generated_imgs_path, exist_ok=True)

    base_count = len(os.listdir(generated_imgs_path))
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
        img_callback=_record_step if record_step_images else None,
        add_caption=print_caption,
        model_weights_path=model_weights_path,
    ):
        prompt = result.prompt
        img_str = ""
        if prompt.init_image:
            img_str = f"_img2img-{prompt.init_image_strength}"
        basefilename = (
            f"{base_count:06}_{prompt.seed}_{prompt.sampler_type.replace('_', '')}{prompt.steps}_"
            f"PS{prompt.prompt_strength}{img_str}_{prompt_normalized(prompt.prompt_text)}"
        )

        for image_type in result.images:
            subpath = os.path.join(outdir, image_type)
            os.makedirs(subpath, exist_ok=True)
            filepath = os.path.join(
                subpath, f"{basefilename}_[{image_type}].{output_file_extension}"
            )
            result.save(filepath, image_type=image_type)
            logger.info(f"🖼  [{image_type}] saved to: {filepath}")
        base_count += 1
        del result


def imagine(
    prompts,
    latent_channels=4,
    downsampling_factor=8,
    precision="autocast",
    img_callback=None,
    half_mode=None,
    add_caption=False,
    model_weights_path=None,
):
    model = load_model(model_weights_location=model_weights_path)

    # only run half-mode on cuda. run it by default
    half_mode = half_mode is None and get_device() == "cuda"
    if half_mode:
        model = model.half()
        # needed when model is in half mode, remove if not using half mode
        # torch.set_default_tensor_type(torch.HalfTensor)
    prompts = [ImaginePrompt(prompts)] if isinstance(prompts, str) else prompts
    prompts = [prompts] if isinstance(prompts, ImaginePrompt) else prompts

    if get_device() == "cpu":
        logger.info("Running in CPU mode. it's gonna be slooooooow.")

    with torch.no_grad(), platform_appropriate_autocast(
        precision
    ), fix_torch_nn_layer_norm(), fix_torch_group_norm():
        for prompt in prompts:
            logger.info(f"Generating 🖼  : {prompt.prompt_description()}")
            with ImageLoggingContext(
                prompt=prompt,
                model=model,
                img_callback=img_callback,
            ):
                seed_everything(prompt.seed)
                model.tile_mode(prompt.tile_mode)

                neutral_conditioning = None
                if prompt.prompt_strength != 1.0:
                    neutral_conditioning = model.get_learned_conditioning(1 * [""])
                    log_conditioning(neutral_conditioning, "neutral conditioning")
                if prompt.conditioning is not None:
                    positive_conditioning = prompt.conditioning
                else:
                    total_weight = sum(wp.weight for wp in prompt.prompts)
                    positive_conditioning = sum(
                        model.get_learned_conditioning(wp.text)
                        * (wp.weight / total_weight)
                        for wp in prompt.prompts
                    )
                log_conditioning(positive_conditioning, "positive conditioning")

                shape = [
                    1,
                    latent_channels,
                    prompt.height // downsampling_factor,
                    prompt.width // downsampling_factor,
                ]

                sampler = get_sampler(prompt.sampler_type, model)
                mask = mask_image = mask_image_orig = mask_grayscale = None
                t_enc = init_latent = init_latent_noised = None
                if prompt.init_image:
                    generation_strength = 1 - prompt.init_image_strength
                    t_enc = int(prompt.steps * generation_strength)
                    try:
                        init_image = pillow_fit_image_within(
                            prompt.init_image,
                            max_height=prompt.height,
                            max_width=prompt.width,
                        )
                    except PIL.UnidentifiedImageError:
                        logger.warning(f"Could not load image: {prompt.init_image}")
                        continue

                    init_image_t = pillow_img_to_torch_image(init_image)

                    if prompt.mask_prompt:
                        mask_image, mask_grayscale = get_img_mask(
                            init_image, prompt.mask_prompt, threshold=0.1
                        )
                    elif prompt.mask_image:
                        mask_image = prompt.mask_image.convert("L")

                    if mask_image is not None:
                        log_img(mask_image, "init mask")
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
                            resample=Image.Resampling.LANCZOS,
                        )
                        log_img(mask_image, "latent_mask")

                        mask = np.array(mask_image)
                        mask = mask.astype(np.float32) / 255.0
                        mask = mask[None, None]
                        mask = torch.from_numpy(mask)
                        mask = mask.to(get_device())

                    init_image_t = init_image_t.to(get_device())
                    init_latent = model.get_first_stage_encoding(
                        model.encode_first_stage(init_image_t)
                    )
                    shape = init_latent.shape

                    log_latent(init_latent, "init_latent")
                    # encode (scaled latent)
                    seed_everything(prompt.seed)
                    noise = randn_seeded(seed=prompt.seed, size=init_latent.size())
                    noise = noise.to(get_device())

                    schedule = NoiseSchedule(
                        model_num_timesteps=model.num_timesteps,
                        ddim_num_steps=prompt.steps,
                        model_alphas_cumprod=model.alphas_cumprod,
                        ddim_discretize="uniform",
                    )
                    if generation_strength >= 1:
                        # prompt strength gets converted to time encodings,
                        # which means you can't get to true 0 without this hack
                        # (or setting steps=1000)
                        init_latent_noised = noise
                    else:
                        init_latent_noised = noise_an_image(
                            init_latent,
                            torch.tensor([t_enc - 1]).to(get_device()),
                            schedule=schedule,
                            noise=noise,
                        )

                log_latent(init_latent_noised, "init_latent_noised")

                samples = sampler.sample(
                    num_steps=prompt.steps,
                    initial_latent=init_latent_noised,
                    positive_conditioning=positive_conditioning,
                    neutral_conditioning=neutral_conditioning,
                    guidance_scale=prompt.prompt_strength,
                    t_start=t_enc,
                    mask=mask,
                    orig_latent=init_latent,
                    shape=shape,
                    batch_size=1,
                )

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = x_sample.to(torch.float32)
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    x_sample_8_orig = x_sample.astype(np.uint8)
                    img = Image.fromarray(x_sample_8_orig)
                    if mask_image_orig and init_image:
                        mask_final = mask_image_orig.filter(
                            ImageFilter.GaussianBlur(radius=3)
                        )
                        log_img(mask_final, "reconstituting mask")
                        mask_final = ImageOps.invert(mask_final)
                        img = Image.composite(img, init_image, mask_final)
                        log_img(img, "reconstituted image")

                    upscaled_img = None
                    rebuilt_orig_img = None

                    if add_caption:
                        caption = generate_caption(img)
                        logger.info(f"Generated caption: {caption}")

                    safety_score = create_safety_score(
                        img,
                        safety_mode=IMAGINAIRY_SAFETY_MODE,
                    )
                    if not safety_score.is_filtered:
                        if prompt.fix_faces:
                            logger.info("Fixing 😊 's in 🖼  using CodeFormer...")
                            img = enhance_faces(img, fidelity=prompt.fix_faces_fidelity)
                        if prompt.upscale:
                            logger.info("Upscaling 🖼  using real-ESRGAN...")
                            upscaled_img = upscale_image(img)

                        # put the newly generated patch back into the original, full size image
                        if (
                            prompt.mask_modify_original
                            and mask_image_orig
                            and prompt.init_image
                        ):
                            img_to_add_back_to_original = (
                                upscaled_img if upscaled_img else img
                            )
                            img_to_add_back_to_original = (
                                img_to_add_back_to_original.resize(
                                    prompt.init_image.size,
                                    resample=Image.Resampling.LANCZOS,
                                )
                            )

                            mask_for_orig_size = mask_image_orig.resize(
                                prompt.init_image.size,
                                resample=Image.Resampling.LANCZOS,
                            )
                            mask_for_orig_size = mask_for_orig_size.filter(
                                ImageFilter.GaussianBlur(radius=5)
                            )
                            log_img(mask_for_orig_size, "mask for original image size")

                            rebuilt_orig_img = Image.composite(
                                prompt.init_image,
                                img_to_add_back_to_original,
                                mask_for_orig_size,
                            )
                            log_img(rebuilt_orig_img, "reconstituted original")

                    yield ImagineResult(
                        img=img,
                        prompt=prompt,
                        upscaled_img=upscaled_img,
                        is_nsfw=safety_score.is_nsfw,
                        safety_score=safety_score,
                        modified_original=rebuilt_orig_img,
                        mask_binary=mask_image_orig,
                        mask_grayscale=mask_grayscale,
                    )


def prompt_normalized(prompt):
    return re.sub(r"[^a-zA-Z0-9.,\[\]-]+", "_", prompt)[:130]

import logging
import os
import re

from imaginairy.schema import ControlNetInput, SafetyMode

logger = logging.getLogger(__name__)

# leave undocumented. I'd ask that no one publicize this flag. Just want a
# slight barrier to entry. Please don't use this is any way that's gonna cause
# the media or politicians to freak out about AI...
IMAGINAIRY_SAFETY_MODE = os.getenv("IMAGINAIRY_SAFETY_MODE", SafetyMode.STRICT)
if IMAGINAIRY_SAFETY_MODE in {"disabled", "classify"}:
    IMAGINAIRY_SAFETY_MODE = SafetyMode.RELAXED
elif IMAGINAIRY_SAFETY_MODE == "filter":
    IMAGINAIRY_SAFETY_MODE = SafetyMode.STRICT

# we put this in the global scope so it can be used in the interactive shell
_most_recent_result = None


def imagine_image_files(
    prompts,
    outdir,
    precision="autocast",
    record_step_images=False,
    output_file_extension="jpg",
    print_caption=False,
    make_gif=False,
    make_compare_gif=False,
    return_filename_type="generated",
):
    from PIL import ImageDraw

    from imaginairy.animations import make_bounce_animation
    from imaginairy.img_utils import pillow_fit_image_within
    from imaginairy.utils import get_next_filenumber

    generated_imgs_path = os.path.join(outdir, "generated")
    os.makedirs(generated_imgs_path, exist_ok=True)

    base_count = get_next_filenumber(generated_imgs_path)
    output_file_extension = output_file_extension.lower()
    if output_file_extension not in {"jpg", "png"}:
        raise ValueError("Must output a png or jpg")

    def _record_step(img, description, image_count, step_count, prompt):
        steps_path = os.path.join(outdir, "steps", f"{base_count:08}_S{prompt.seed}")
        os.makedirs(steps_path, exist_ok=True)
        filename = f"{base_count:08}_S{prompt.seed}_{image_count:04}_step{step_count:03}_{prompt_normalized(description)[:40]}.jpg"

        destination = os.path.join(steps_path, filename)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), str(description))
        img.save(destination)

    if make_gif:
        for p in prompts:
            p.collect_progress_latents = True
    result_filenames = []
    for result in imagine(
        prompts,
        precision=precision,
        debug_img_callback=_record_step if record_step_images else None,
        add_caption=print_caption,
    ):
        prompt = result.prompt
        if prompt.is_intermediate:
            # we don't save intermediate images
            continue
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
            logger.info(f"    [{image_type}] saved to: {filepath}")
            if image_type == return_filename_type:
                result_filenames.append(filepath)

        if make_gif and result.progress_latents:
            subpath = os.path.join(outdir, "gif")
            os.makedirs(subpath, exist_ok=True)
            filepath = os.path.join(subpath, f"{basefilename}.gif")

            frames = [*result.progress_latents, result.images["generated"]]

            if prompt.init_image:
                resized_init_image = pillow_fit_image_within(
                    prompt.init_image, prompt.width, prompt.height
                )
                frames = [resized_init_image, *frames]
            frames.reverse()
            make_bounce_animation(
                imgs=frames,
                outpath=filepath,
                start_pause_duration_ms=1500,
                end_pause_duration_ms=1000,
            )
            logger.info(f"    [gif] {len(frames)} frames saved to: {filepath}")
        if make_compare_gif and prompt.init_image:
            subpath = os.path.join(outdir, "gif")
            os.makedirs(subpath, exist_ok=True)
            filepath = os.path.join(subpath, f"{basefilename}_[compare].gif")
            resized_init_image = pillow_fit_image_within(
                prompt.init_image, prompt.width, prompt.height
            )
            frames = [result.images["generated"], resized_init_image]

            make_bounce_animation(
                imgs=frames,
                outpath=filepath,
            )
            logger.info(f"    [gif-comparison] saved to: {filepath}")

        base_count += 1
        del result

    return result_filenames


def imagine(
    prompts,
    precision="autocast",
    debug_img_callback=None,
    progress_img_callback=None,
    progress_img_interval_steps=3,
    progress_img_interval_min_s=0.1,
    half_mode=None,
    add_caption=False,
    unsafe_retry_count=1,
):
    import torch.nn

    from imaginairy.api_refiners import _generate_single_image
    from imaginairy.schema import ImaginePrompt
    from imaginairy.utils import (
        check_torch_version,
        fix_torch_group_norm,
        fix_torch_nn_layer_norm,
        get_device,
        platform_appropriate_autocast,
    )

    check_torch_version()

    prompts = [ImaginePrompt(prompts)] if isinstance(prompts, str) else prompts
    prompts = [prompts] if isinstance(prompts, ImaginePrompt) else prompts

    try:
        num_prompts = str(len(prompts))
    except TypeError:
        num_prompts = "?"

    if get_device() == "cpu":
        logger.info("Running in CPU mode. it's gonna be slooooooow.")

    with torch.no_grad(), platform_appropriate_autocast(
        precision
    ), fix_torch_nn_layer_norm(), fix_torch_group_norm():
        for i, prompt in enumerate(prompts):
            logger.info(
                f"🖼  Generating  {i + 1}/{num_prompts}: {prompt.prompt_description()}"
            )
            for attempt in range(unsafe_retry_count + 1):
                if attempt > 0 and isinstance(prompt.seed, int):
                    prompt.seed += 100_000_000 + attempt
                result = _generate_single_image(
                    prompt,
                    debug_img_callback=debug_img_callback,
                    progress_img_callback=progress_img_callback,
                    progress_img_interval_steps=progress_img_interval_steps,
                    progress_img_interval_min_s=progress_img_interval_min_s,
                    half_mode=half_mode,
                    add_caption=add_caption,
                )
                if not result.safety_score.is_filtered:
                    break
                if attempt < unsafe_retry_count:
                    logger.info("    Image was unsafe, retrying with new seed...")

            yield result


def _generate_single_image_compvis(
    prompt,
    debug_img_callback=None,
    progress_img_callback=None,
    progress_img_interval_steps=3,
    progress_img_interval_min_s=0.1,
    half_mode=None,
    add_caption=False,
    # controlnet, finetune, naive, auto
    inpaint_method="finetune",
    return_latent=False,
):
    import torch.nn
    from PIL import Image, ImageOps
    from pytorch_lightning import seed_everything

    from imaginairy.enhancers.clip_masking import get_img_mask
    from imaginairy.enhancers.describe_image_blip import generate_caption
    from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
    from imaginairy.enhancers.upscale_realesrgan import upscale_image
    from imaginairy.img_utils import (
        add_caption_to_image,
        pillow_fit_image_within,
        pillow_img_to_torch_image,
        pillow_mask_to_latent_mask,
        torch_img_to_pillow_img,
    )
    from imaginairy.log_utils import (
        ImageLoggingContext,
        log_conditioning,
        log_img,
        log_latent,
    )
    from imaginairy.model_manager import (
        get_diffusion_model,
        get_model_default_image_size,
    )
    from imaginairy.modules.midas.api import torch_image_to_depth_map
    from imaginairy.outpaint import outpaint_arg_str_parse, prepare_image_for_outpaint
    from imaginairy.safety import create_safety_score
    from imaginairy.samplers import SAMPLER_LOOKUP
    from imaginairy.samplers.editing import CFGEditingDenoiser
    from imaginairy.schema import ImaginePrompt, ImagineResult
    from imaginairy.utils import get_device, randn_seeded

    latent_channels = 4
    downsampling_factor = 8
    batch_size = 1
    global _most_recent_result
    # handle prompt pulling in previous values
    # if isinstance(prompt.init_image, str) and prompt.init_image.startswith("*prev"):
    #     _, img_type = prompt.init_image.strip("*").split(".")
    #     prompt.init_image = _most_recent_result.images[img_type]
    # if isinstance(prompt.mask_image, str) and prompt.mask_image.startswith("*prev"):
    #     _, img_type = prompt.mask_image.strip("*").split(".")
    #     prompt.mask_image = _most_recent_result.images[img_type]
    prompt = prompt.make_concrete_copy()

    control_modes = []
    control_inputs = prompt.control_inputs or []
    control_inputs = control_inputs.copy()
    for_inpainting = bool(prompt.mask_image or prompt.mask_prompt or prompt.outpaint)

    if control_inputs:
        control_modes = [c.mode for c in prompt.control_inputs]
    if inpaint_method == "auto":
        if prompt.model in {"SD-1.5", "SD-2.0"}:
            inpaint_method = "finetune"
        else:
            inpaint_method = "controlnet"

    if for_inpainting and inpaint_method == "controlnet":
        control_modes.append("inpaint")
    model = get_diffusion_model(
        weights_location=prompt.model,
        config_path=prompt.model_config_path,
        control_weights_locations=control_modes,
        half_mode=half_mode,
        for_inpainting=for_inpainting and inpaint_method == "finetune",
    )
    is_controlnet_model = hasattr(model, "control_key")

    progress_latents = []

    def latent_logger(latents):
        progress_latents.append(latents)

    with ImageLoggingContext(
        prompt=prompt,
        model=model,
        debug_img_callback=debug_img_callback,
        progress_img_callback=progress_img_callback,
        progress_img_interval_steps=progress_img_interval_steps,
        progress_img_interval_min_s=progress_img_interval_min_s,
        progress_latent_callback=latent_logger
        if prompt.collect_progress_latents
        else None,
    ) as lc:
        seed_everything(prompt.seed)

        model.tile_mode(prompt.tile_mode)
        with lc.timing("conditioning"):
            # need to expand if doing batches
            neutral_conditioning = _prompts_to_embeddings(prompt.negative_prompt, model)
            _prompts_to_embeddings("", model)
            log_conditioning(neutral_conditioning, "neutral conditioning")
            if prompt.conditioning is not None:
                positive_conditioning = prompt.conditioning
            else:
                positive_conditioning = _prompts_to_embeddings(prompt.prompts, model)
            log_conditioning(positive_conditioning, "positive conditioning")

        shape = [
            batch_size,
            latent_channels,
            prompt.height // downsampling_factor,
            prompt.width // downsampling_factor,
        ]
        SamplerCls = SAMPLER_LOOKUP[prompt.sampler_type.lower()]
        sampler = SamplerCls(model)
        mask_latent = mask_image = mask_image_orig = mask_grayscale = None
        t_enc = init_latent = control_image = None
        starting_image = None
        denoiser_cls = None

        c_cat = []
        c_cat_neutral = None
        result_images = {}
        seed_everything(prompt.seed)
        noise = randn_seeded(seed=prompt.seed, size=shape).to(get_device())
        control_strengths = []

        if prompt.init_image:
            starting_image = prompt.init_image
            generation_strength = 1 - prompt.init_image_strength

            if model.cond_stage_key == "edit" or generation_strength >= 1:
                t_enc = None
            else:
                t_enc = int(prompt.steps * generation_strength)

            if prompt.mask_prompt:
                mask_image, mask_grayscale = get_img_mask(
                    starting_image, prompt.mask_prompt, threshold=0.1
                )
            elif prompt.mask_image:
                mask_image = prompt.mask_image.convert("L")
            if prompt.outpaint:
                outpaint_kwargs = outpaint_arg_str_parse(prompt.outpaint)
                starting_image, mask_image = prepare_image_for_outpaint(
                    starting_image, mask_image, **outpaint_kwargs
                )

            init_image = pillow_fit_image_within(
                starting_image,
                max_height=prompt.height,
                max_width=prompt.width,
            )
            init_image_t = pillow_img_to_torch_image(init_image)
            init_image_t = init_image_t.to(get_device())
            init_latent = model.get_first_stage_encoding(
                model.encode_first_stage(init_image_t)
            )
            shape = init_latent.shape

            log_latent(init_latent, "init_latent")

            if mask_image is not None:
                mask_image = pillow_fit_image_within(
                    mask_image,
                    max_height=prompt.height,
                    max_width=prompt.width,
                    convert="L",
                )

                log_img(mask_image, "init mask")

                if prompt.mask_mode == ImaginePrompt.MaskMode.REPLACE:
                    mask_image = ImageOps.invert(mask_image)

                mask_image_orig = mask_image
                log_img(mask_image, "latent_mask")
                mask_latent = pillow_mask_to_latent_mask(
                    mask_image, downsampling_factor=downsampling_factor
                ).to(get_device())
                if inpaint_method == "controlnet":
                    result_images["control-inpaint"] = mask_image
                    control_inputs.append(
                        ControlNetInput(mode="inpaint", image=mask_image)
                    )

            seed_everything(prompt.seed)
            noise = randn_seeded(seed=prompt.seed, size=init_latent.shape).to(
                get_device()
            )
            # noise = noise[:, :, : init_latent.shape[2], : init_latent.shape[3]]

            # schedule = NoiseSchedule(
            #     model_num_timesteps=model.num_timesteps,
            #     ddim_num_steps=prompt.steps,
            #     model_alphas_cumprod=model.alphas_cumprod,
            #     ddim_discretize="uniform",
            # )
            # if generation_strength >= 1:
            #     # prompt strength gets converted to time encodings,
            #     # which means you can't get to true 0 without this hack
            #     # (or setting steps=1000)
            #     init_latent_noised = noise
            # else:
            #     init_latent_noised = noise_an_image(
            #         init_latent,
            #         torch.tensor([t_enc - 1]).to(get_device()),
            #         schedule=schedule,
            #         noise=noise,
            #     )

        if hasattr(model, "depth_stage_key"):
            # depth model
            depth_t = torch_image_to_depth_map(init_image_t)
            depth_latent = torch.nn.functional.interpolate(
                depth_t,
                size=shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            result_images["depth_image"] = depth_t
            c_cat.append(depth_latent)

        elif is_controlnet_model:
            from imaginairy.img_processors.control_modes import CONTROL_MODES

            for control_input in control_inputs:
                if control_input.image_raw is not None:
                    control_image = control_input.image_raw
                elif control_input.image is not None:
                    control_image = control_input.image
                control_image = control_image.convert("RGB")
                log_img(control_image, "control_image_input")
                control_image_input = pillow_fit_image_within(
                    control_image,
                    max_height=prompt.height,
                    max_width=prompt.width,
                )
                control_image_input_t = pillow_img_to_torch_image(control_image_input)
                control_image_input_t = control_image_input_t.to(get_device())

                if control_input.image_raw is None:
                    control_prep_function = CONTROL_MODES[control_input.mode]
                    if control_input.mode == "inpaint":
                        control_image_t = control_prep_function(
                            control_image_input_t, init_image_t
                        )
                    else:
                        control_image_t = control_prep_function(control_image_input_t)
                else:
                    control_image_t = (control_image_input_t + 1) / 2

                control_image_disp = control_image_t * 2 - 1
                result_images[f"control-{control_input.mode}"] = control_image_disp
                log_img(control_image_disp, "control_image")

                if len(control_image_t.shape) == 3:
                    raise RuntimeError("Control image must be 4D")

                if control_image_t.shape[1] != 3:
                    raise RuntimeError("Control image must have 3 channels")

                if (
                    control_input.mode != "inpaint"
                    and control_image_t.min() < 0
                    or control_image_t.max() > 1
                ):
                    msg = f"Control image must be in [0, 1] but we received {control_image_t.min()} and {control_image_t.max()}"
                    raise RuntimeError(msg)

                if control_image_t.max() == control_image_t.min():
                    msg = f"No control signal found in control image {control_input.mode}."
                    raise RuntimeError(msg)

                c_cat.append(control_image_t)
                control_strengths.append(control_input.strength)

        elif hasattr(model, "masked_image_key"):
            # inpainting model
            mask_t = pillow_img_to_torch_image(ImageOps.invert(mask_image_orig)).to(
                get_device()
            )
            inverted_mask = 1 - mask_latent
            masked_image_t = init_image_t * (mask_t < 0.5)
            log_img(masked_image_t, "masked_image")

            inverted_mask_latent = torch.nn.functional.interpolate(
                inverted_mask, size=shape[-2:]
            )
            c_cat.append(inverted_mask_latent)

            masked_image_latent = model.get_first_stage_encoding(
                model.encode_first_stage(masked_image_t)
            )
            c_cat.append(masked_image_latent)

        elif model.cond_stage_key == "edit":
            # pix2pix model
            c_cat = [model.encode_first_stage(init_image_t)]
            c_cat_neutral = [torch.zeros_like(init_latent)]
            denoiser_cls = CFGEditingDenoiser
        if c_cat:
            c_cat = [torch.cat([c], dim=1) for c in c_cat]

        if c_cat_neutral is None:
            c_cat_neutral = c_cat

        positive_conditioning = {
            "c_concat": c_cat,
            "c_crossattn": [positive_conditioning],
        }
        neutral_conditioning = {
            "c_concat": c_cat_neutral,
            "c_crossattn": [neutral_conditioning],
        }

        if control_strengths and is_controlnet_model:
            positive_conditioning["control_strengths"] = torch.Tensor(control_strengths)
            neutral_conditioning["control_strengths"] = torch.Tensor(control_strengths)

        if (
            prompt.allow_compose_phase
            and not is_controlnet_model
            and model.cond_stage_key != "edit"
        ):
            if prompt.init_image:
                comp_image = _generate_composition_image(
                    prompt=prompt,
                    target_height=init_image.height,
                    target_width=init_image.width,
                    cutoff=get_model_default_image_size(prompt.model),
                )
            else:
                comp_image = _generate_composition_image(
                    prompt=prompt,
                    target_height=prompt.height,
                    target_width=prompt.width,
                    cutoff=get_model_default_image_size(prompt.model),
                )
            if comp_image is not None:
                result_images["composition"] = comp_image
                # noise = noise[:, :, : comp_image.height, : comp_image.shape[3]]
                t_enc = int(prompt.steps * 0.65)
                log_img(comp_image, "comp_image")
                comp_image_t = pillow_img_to_torch_image(comp_image)
                comp_image_t = comp_image_t.to(get_device())
                init_latent = model.get_first_stage_encoding(
                    model.encode_first_stage(comp_image_t)
                )
        with lc.timing("sampling"):
            samples = sampler.sample(
                num_steps=prompt.steps,
                positive_conditioning=positive_conditioning,
                neutral_conditioning=neutral_conditioning,
                guidance_scale=prompt.prompt_strength,
                t_start=t_enc,
                mask=mask_latent,
                orig_latent=init_latent,
                shape=shape,
                batch_size=1,
                denoiser_cls=denoiser_cls,
                noise=noise,
            )
        if return_latent:
            return samples

        with lc.timing("decoding"):
            gen_imgs_t = model.decode_first_stage(samples)
            gen_img = torch_img_to_pillow_img(gen_imgs_t)

        if mask_image_orig and init_image:
            mask_final = mask_image_orig.copy()
            log_img(mask_final, "reconstituting mask")
            mask_final = ImageOps.invert(mask_final)
            gen_img = Image.composite(gen_img, init_image, mask_final)
            gen_img = combine_image(
                original_img=init_image,
                generated_img=gen_img,
                mask_img=mask_image_orig,
            )
            log_img(gen_img, "reconstituted image")

        upscaled_img = None
        rebuilt_orig_img = None

        if add_caption:
            caption = generate_caption(gen_img)
            logger.info(f"Generated caption: {caption}")

        with lc.timing("safety-filter"):
            safety_score = create_safety_score(
                gen_img,
                safety_mode=IMAGINAIRY_SAFETY_MODE,
            )
        if safety_score.is_filtered:
            progress_latents.clear()
        if not safety_score.is_filtered:
            if prompt.fix_faces:
                logger.info("Fixing 😊 's in 🖼  using CodeFormer...")
                with lc.timing("face enhancement"):
                    gen_img = enhance_faces(gen_img, fidelity=prompt.fix_faces_fidelity)
            if prompt.upscale:
                logger.info("Upscaling 🖼  using real-ESRGAN...")
                with lc.timing("upscaling"):
                    upscaled_img = upscale_image(gen_img)

            # put the newly generated patch back into the original, full-size image
            if prompt.mask_modify_original and mask_image_orig and starting_image:
                img_to_add_back_to_original = upscaled_img if upscaled_img else gen_img
                rebuilt_orig_img = combine_image(
                    original_img=starting_image,
                    generated_img=img_to_add_back_to_original,
                    mask_img=mask_image_orig,
                )

            if prompt.caption_text:
                caption_text = prompt.caption_text.format(prompt=prompt.prompt_text)
                add_caption_to_image(gen_img, caption_text)

        result = ImagineResult(
            img=gen_img,
            prompt=prompt,
            upscaled_img=upscaled_img,
            is_nsfw=safety_score.is_nsfw,
            safety_score=safety_score,
            modified_original=rebuilt_orig_img,
            mask_binary=mask_image_orig,
            mask_grayscale=mask_grayscale,
            result_images=result_images,
            timings=lc.get_timings(),
            progress_latents=progress_latents.copy(),
        )

        _most_recent_result = result
        logger.info(f"Image Generated. Timings: {result.timings_str()}")
        return result


def _prompts_to_embeddings(prompts, model):
    total_weight = sum(wp.weight for wp in prompts)
    conditioning = sum(
        model.get_learned_conditioning(wp.text) * (wp.weight / total_weight)
        for wp in prompts
    )
    return conditioning


def calc_scale_to_fit_within(
    height,
    width,
    max_size,
):
    if max(height, width) < max_size:
        return 1

    if width > height:
        return max_size / width

    return max_size / height


def _scale_latent(
    latent,
    model,
    h,
    w,
):
    from torch.nn import functional as F

    # convert to non-latent-space first
    img = model.decode_first_stage(latent)
    img = F.interpolate(img, size=(h, w), mode="bicubic", align_corners=False)
    latent = model.get_first_stage_encoding(model.encode_first_stage(img))
    return latent


def _generate_composition_image(prompt, target_height, target_width, cutoff=512):
    from PIL import Image

    from imaginairy.api_refiners import _generate_single_image

    if prompt.width <= cutoff and prompt.height <= cutoff:
        return None, None

    shrink_scale = calc_scale_to_fit_within(
        height=prompt.height,
        width=prompt.width,
        max_size=cutoff,
    )

    composition_prompt = prompt.full_copy(
        deep=True,
        update={
            "width": int(prompt.width * shrink_scale),
            "height": int(prompt.height * shrink_scale),
            "steps": None,
            "upscale": False,
            "fix_faces": False,
            "mask_modify_original": False,
        },
    )

    result = _generate_single_image(composition_prompt)
    img = result.images["generated"]
    while img.width < target_width:
        from imaginairy.enhancers.upscale_realesrgan import upscale_image

        img = upscale_image(img)

    # samples = _generate_single_image(composition_prompt, return_latent=True)
    # while samples.shape[-1] * 8 < target_width:
    #     samples = upscale_latent(samples)
    #
    # img = model_latent_to_pillow_img(samples)

    img = img.resize(
        (target_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )

    return img, result.images["generated"]


def prompt_normalized(prompt, length=130):
    return re.sub(r"[^a-zA-Z0-9.,\[\]-]+", "_", prompt)[:length]


def combine_image(original_img, generated_img, mask_img):
    """Combine the generated image with the original image using the mask image."""
    from PIL import Image

    from imaginairy.log_utils import log_img

    generated_img = generated_img.resize(
        original_img.size,
        resample=Image.Resampling.LANCZOS,
    )

    mask_for_orig_size = mask_img.resize(
        original_img.size,
        resample=Image.Resampling.LANCZOS,
    )
    log_img(mask_for_orig_size, "mask for original image size")

    rebuilt_orig_img = Image.composite(
        original_img,
        generated_img,
        mask_for_orig_size,
    )
    log_img(rebuilt_orig_img, "reconstituted original")
    return rebuilt_orig_img

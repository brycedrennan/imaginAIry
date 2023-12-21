import logging
from typing import TYPE_CHECKING, Any

from imaginairy.api.generate import IMAGINAIRY_SAFETY_MODE
from imaginairy.utils.img_utils import calc_scale_to_fit_within, combine_image
from imaginairy.utils.named_resolutions import normalize_image_size

if TYPE_CHECKING:
    from imaginairy.schema import ImaginePrompt

logger = logging.getLogger(__name__)


def _generate_single_image_compvis(
    prompt: "ImaginePrompt",
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
    from imaginairy.modules.midas.api import torch_image_to_depth_map
    from imaginairy.samplers import SOLVER_LOOKUP
    from imaginairy.samplers.editing import CFGEditingDenoiser
    from imaginairy.schema import (
        ControlInput,
        ImagineResult,
        LazyLoadingImage,
        MaskMode,
    )
    from imaginairy.utils import get_device, randn_seeded
    from imaginairy.utils.img_utils import (
        add_caption_to_image,
        pillow_fit_image_within,
        pillow_img_to_torch_image,
        pillow_mask_to_latent_mask,
        torch_img_to_pillow_img,
    )
    from imaginairy.utils.log_utils import (
        ImageLoggingContext,
        log_conditioning,
        log_img,
        log_latent,
    )
    from imaginairy.utils.model_manager import (
        get_diffusion_model,
        get_model_default_image_size,
    )
    from imaginairy.utils.outpaint import (
        outpaint_arg_str_parse,
        prepare_image_for_outpaint,
    )
    from imaginairy.utils.safety import create_safety_score

    latent_channels = 4
    downsampling_factor = 8
    batch_size = 1
    # global _most_recent_result
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
        if prompt.model_weights in {"SD-1.5"}:
            inpaint_method = "finetune"
        else:
            inpaint_method = "controlnet"

    if for_inpainting and inpaint_method == "controlnet":
        control_modes.append("inpaint")
    model = get_diffusion_model(
        weights_location=prompt.model_weights,
        config_path=prompt.model_architecture,
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
        SolverCls = SOLVER_LOOKUP[prompt.solver_type.lower()]
        solver = SolverCls(model)
        mask_image: Image.Image | LazyLoadingImage | None = None
        mask_latent = mask_image_orig = mask_grayscale = None
        init_latent: torch.Tensor | None = None
        t_enc = None
        starting_image = None
        denoiser_cls = None

        c_cat = []
        c_cat_neutral = None
        result_images: dict[str, torch.Tensor | Image.Image | None] = {}
        assert prompt.seed is not None
        seed_everything(prompt.seed)
        noise = randn_seeded(seed=prompt.seed, size=shape).to(get_device())
        control_strengths = []

        if prompt.init_image:
            starting_image = prompt.init_image
            assert prompt.init_image_strength is not None
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
            assert starting_image is not None
            init_image = pillow_fit_image_within(
                starting_image,
                max_height=prompt.height,
                max_width=prompt.width,
            )
            init_image_t = pillow_img_to_torch_image(init_image).to(get_device())
            init_latent = model.get_first_stage_encoding(
                model.encode_first_stage(init_image_t)
            )
            assert init_latent is not None
            shape = list(init_latent.shape)

            log_latent(init_latent, "init_latent")

            if mask_image is not None:
                mask_image = pillow_fit_image_within(
                    mask_image,
                    max_height=prompt.height,
                    max_width=prompt.width,
                    convert="L",
                )

                log_img(mask_image, "init mask")

                if prompt.mask_mode == MaskMode.REPLACE:
                    mask_image = ImageOps.invert(mask_image)

                mask_image_orig = mask_image
                log_img(mask_image, "latent_mask")
                mask_latent = pillow_mask_to_latent_mask(
                    mask_image, downsampling_factor=downsampling_factor
                ).to(get_device())
                if inpaint_method == "controlnet":
                    result_images["control-inpaint"] = mask_image
                    control_inputs.append(
                        ControlInput(mode="inpaint", image=mask_image)
                    )
            assert prompt.seed is not None
            seed_everything(prompt.seed)
            noise = randn_seeded(seed=prompt.seed, size=list(init_latent.shape)).to(
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
                else:
                    raise RuntimeError("Control image must be provided")
                assert control_image is not None
                control_image = control_image.convert("RGB")
                log_img(control_image, "control_image_input")
                assert control_image is not None

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
                        control_image_t = control_prep_function(  # type: ignore
                            control_image_input_t, init_image_t
                        )
                    else:
                        control_image_t = control_prep_function(control_image_input_t)  # type: ignore
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
            assert mask_image_orig is not None
            assert mask_latent is not None
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
            assert init_latent is not None
            c_cat_neutral = [torch.zeros_like(init_latent)]
            denoiser_cls = CFGEditingDenoiser
        if c_cat:
            c_cat = [torch.cat([c], dim=1) for c in c_cat]

        if c_cat_neutral is None:
            c_cat_neutral = c_cat

        positive_conditioning_d: dict[str, Any] = {
            "c_concat": c_cat,
            "c_crossattn": [positive_conditioning],
        }
        neutral_conditioning_d: dict[str, Any] = {
            "c_concat": c_cat_neutral,
            "c_crossattn": [neutral_conditioning],
        }
        del neutral_conditioning
        del positive_conditioning

        if control_strengths and is_controlnet_model:
            positive_conditioning_d["control_strengths"] = torch.Tensor(
                control_strengths
            )
            neutral_conditioning_d["control_strengths"] = torch.Tensor(
                control_strengths
            )

        if (
            prompt.allow_compose_phase
            and not is_controlnet_model
            and model.cond_stage_key != "edit"
        ):
            default_size = get_model_default_image_size(
                prompt.model_weights.architecture
            )
            if prompt.init_image:
                comp_image = _generate_composition_image(
                    prompt=prompt,
                    target_height=init_image.height,
                    target_width=init_image.width,
                    cutoff=default_size,
                )
            else:
                comp_image = _generate_composition_image(
                    prompt=prompt,
                    target_height=prompt.height,
                    target_width=prompt.width,
                    cutoff=default_size,
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
            samples = solver.sample(
                num_steps=prompt.steps,
                positive_conditioning=positive_conditioning_d,
                neutral_conditioning=neutral_conditioning_d,
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

        result_images["upscaled"] = upscaled_img
        result_images["modified_original"] = rebuilt_orig_img
        result_images["mask_binary"] = mask_image_orig
        result_images["mask_grayscale"] = mask_grayscale

        result = ImagineResult(
            img=gen_img,
            prompt=prompt,
            is_nsfw=safety_score.is_nsfw,
            safety_score=safety_score,
            result_images=result_images,
            timings=lc.get_timings(),
            progress_latents=progress_latents.copy(),
        )

        # _most_recent_result = result
        logger.info(f"Image Generated. Timings: {result.timings_str()}")
        return result


def _prompts_to_embeddings(prompts, model):
    total_weight = sum(wp.weight for wp in prompts)
    conditioning = sum(
        model.get_learned_conditioning(wp.text) * (wp.weight / total_weight)
        for wp in prompts
    )
    return conditioning


def _generate_composition_image(
    prompt,
    target_height,
    target_width,
    cutoff: tuple[int, int] = (512, 512),
    dtype=None,
):
    from PIL import Image

    from imaginairy.api.generate_refiners import generate_single_image
    from imaginairy.utils import default, get_default_dtype

    cutoff = normalize_image_size(cutoff)
    if prompt.width <= cutoff[0] and prompt.height <= cutoff[1]:
        return None, None

    dtype = default(dtype, get_default_dtype)

    shrink_scale = calc_scale_to_fit_within(
        height=prompt.height,
        width=prompt.width,
        max_size=cutoff,
    )

    composition_prompt = prompt.full_copy(
        deep=True,
        update={
            "size": (
                int(prompt.width * shrink_scale),
                int(prompt.height * shrink_scale),
            ),
            "steps": None,
            "upscale": False,
            "fix_faces": False,
            "mask_modify_original": False,
            "allow_compose_phase": False,
        },
    )

    result = generate_single_image(composition_prompt, dtype=dtype)
    img = result.images["generated"]
    while img.width < target_width:
        from imaginairy.enhancers.upscale_realesrgan import upscale_image

        img = upscale_image(img)

    # samples = generate_single_image(composition_prompt, return_latent=True)
    # while samples.shape[-1] * 8 < target_width:
    #     samples = upscale_latent(samples)
    #
    # img = model_latent_to_pillow_img(samples)

    img = img.resize(
        (target_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )

    return img, result.images["generated"]

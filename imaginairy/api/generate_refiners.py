"""Functions for generating refined images"""

import logging
from contextlib import nullcontext
from typing import Any

from imaginairy.config import CONTROL_CONFIG_SHORTCUTS
from imaginairy.schema import ControlInput, ImaginePrompt, MaskMode
from imaginairy.utils import clear_gpu_cache, seed_everything
from imaginairy.utils.log_utils import ImageLoggingContext

logger = logging.getLogger(__name__)

MULTIDIFFUSION_TILE_SIZE = 128  # latent units (1024px)
MULTIDIFFUSION_STRIDE = 64  # 50% overlap


def _get_tile_positions(size: int, tile_size: int, stride: int) -> list[int]:
    """Return tile start positions covering `size` with given tile/stride.

    Last position snaps to `size - tile_size` to ensure full coverage.
    """
    if size <= tile_size:
        return [0]
    positions = list(range(0, size - tile_size, stride))
    last = size - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _caption_tile_regions(
    comp_image,
    tile_ys: list[int],
    tile_xs: list[int],
    tile_size: int,
    downsampling_factor: int,
    global_prompts,
    negative_prompts,
    sd,
):
    """Caption each tile region of comp_image with BLIP; return per-tile conditioning.

    Deduplicates identical captions so `calculate_text_conditioning_kwargs` is
    called at most once per unique caption string.
    """
    from imaginairy.enhancers.describe_image_blip import generate_caption
    from imaginairy.schema import WeightedPrompt

    px = downsampling_factor
    caption_to_kwargs: dict[str, dict] = {}
    tile_conditioning: dict[tuple[int, int], dict] = {}

    for ty in tile_ys:
        for tx in tile_xs:
            crop_box = (
                tx * px,
                ty * px,
                (tx + tile_size) * px,
                (ty + tile_size) * px,
            )
            crop = comp_image.crop(crop_box)
            caption = generate_caption(crop, min_length=5)

            if caption not in caption_to_kwargs:
                global_text = " ".join(wp.text for wp in global_prompts)
                combined = f"{global_text}, {caption}"
                kwargs = sd.calculate_text_conditioning_kwargs(
                    positive_prompts=[WeightedPrompt(text=combined)],
                    negative_prompts=negative_prompts,
                )
                for k, v in kwargs.items():
                    kwargs[k] = v.to(device=sd.unet.device, dtype=sd.unet.dtype)
                caption_to_kwargs[caption] = kwargs

            tile_conditioning[(ty, tx)] = caption_to_kwargs[caption]

    unique = len(caption_to_kwargs)
    total = len(tile_ys) * len(tile_xs)
    logger.info(f"Per-tile captioning: {total} tiles, {unique} unique captions")
    return tile_conditioning


def generate_single_image(
    prompt: ImaginePrompt,
    debug_img_callback=None,
    progress_img_callback=None,
    progress_img_interval_steps=3,
    progress_img_interval_min_s=0.1,
    add_caption=False,
    return_latent=False,
    dtype=None,
    logging_context: ImageLoggingContext | None = None,
    output_perf=False,
    image_name="",
):
    import torch.nn
    from PIL import Image, ImageOps
    from tqdm import tqdm

    from imaginairy.api.generate import (
        IMAGINAIRY_SAFETY_MODE,
    )
    from imaginairy.enhancers.clip_masking import get_img_mask
    from imaginairy.enhancers.describe_image_blip import generate_caption
    from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
    from imaginairy.enhancers.upscale import upscale_image
    from imaginairy.samplers import SolverName
    from imaginairy.schema import ImagineResult
    from imaginairy.utils import get_device, randn_seeded
    from imaginairy.utils.img_utils import (
        add_caption_to_image,
        combine_image,
        pillow_fit_image_within,
        pillow_img_to_torch_image,
        pillow_mask_to_latent_mask,
    )
    from imaginairy.utils.log_utils import (
        ImageLoggingContext,
        log_img,
        log_latent,
    )
    from imaginairy.utils.model_manager import (
        get_diffusion_model_refiners,
        get_model_default_image_size,
    )
    from imaginairy.utils.outpaint import (
        outpaint_arg_str_parse,
        prepare_image_for_outpaint,
    )
    from imaginairy.utils.safety import create_safety_score
    from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers import (
        DDIM,
        DPMSolver,
    )

    if dtype is None:
        dtype = torch.float16

    get_device()
    clear_gpu_cache()
    prompt = prompt.make_concrete_copy()

    if not logging_context:

        def latent_logger(latents):
            progress_latents.append(latents)

        lc = ImageLoggingContext(
            prompt=prompt,
            debug_img_callback=debug_img_callback,
            progress_img_callback=progress_img_callback,
            progress_img_interval_steps=progress_img_interval_steps,
            progress_img_interval_min_s=progress_img_interval_min_s,
            progress_latent_callback=latent_logger
            if prompt.collect_progress_latents
            else None,
        )
        _context: Any = lc
    else:
        lc = logging_context
        _context = nullcontext()
    with _context:
        with lc.timing("model-load"):
            sd = get_diffusion_model_refiners(
                weights_config=prompt.model_weights,
                for_inpainting=prompt.should_use_inpainting
                and prompt.inpaint_method == "finetune",
                dtype=dtype,
            )
            if prompt.inpaint_method == "patch" and prompt.should_use_inpainting:
                from imaginairy.modules.fooocus_inpaint import (
                    apply_fooocus_patch_to_unet,
                )

                with lc.timing("fooocus-patch"):
                    apply_fooocus_patch_to_unet(sd.unet)
        lc.model = sd
        seed_everything(prompt.seed)
        downsampling_factor = 8
        latent_channels = 4
        batch_size = 1

        mask_image = None
        mask_image_orig = None

        sd.set_tile_mode(prompt.tile_mode)

        result_images: dict[str, torch.Tensor | None | Image.Image] = {}
        progress_latents: list[torch.Tensor] = []
        first_step = 0
        mask_grayscale = None
        tile_conditioning: dict[tuple[int, int], dict] = {}

        shape = [
            batch_size,
            latent_channels,
            prompt.height // downsampling_factor,
            prompt.width // downsampling_factor,
        ]

        init_latent = None
        noise_step = None
        _patch_latent_mask = None  # set when using fooocus patch inpainting

        control_modes = []
        control_inputs = prompt.control_inputs or []
        control_inputs = control_inputs.copy()

        if control_inputs:
            control_modes = [c.mode for c in prompt.control_inputs]

        if prompt.init_image:
            starting_image = prompt.init_image
            assert prompt.init_image_strength is not None
            first_step = int(prompt.steps * prompt.init_image_strength)
            # noise_step = int((prompt.steps - 1) * prompt.init_image_strength)

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
            init_image_t = init_image_t.to(device=sd.lda.device, dtype=sd.lda.dtype)
            init_latent = sd.lda.encode(init_image_t)

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
                pillow_mask_to_latent_mask(
                    mask_image, downsampling_factor=downsampling_factor
                ).to(get_device())
                if prompt.inpaint_method == "controlnet":
                    result_images["control-inpaint"] = mask_image
                    control_inputs.append(
                        ControlInput(mode="inpaint", image=mask_image)
                    )

        seed_everything(prompt.seed)
        assert prompt.seed is not None

        noise = randn_seeded(seed=prompt.seed, size=shape).to(
            sd.unet.device, dtype=sd.unet.dtype
        )
        noised_latent = noise
        controlnets = []

        if control_modes:
            with lc.timing("control-image-prep"):
                for control_input in control_inputs:
                    (
                        controlnet,
                        control_image_t,
                        control_image_disp,
                    ) = prep_control_input(
                        control_input=control_input,
                        sd=sd,
                        init_image_t=init_image_t,
                        fit_width=prompt.width,
                        fit_height=prompt.height,
                    )
                    result_images[f"control-{control_input.mode}"] = control_image_disp
                    controlnets.append((controlnet, control_image_t))

        if prompt.allow_compose_phase:
            with lc.timing("composition"):
                cutoff_size = get_model_default_image_size(prompt.model_architecture)
                cutoff_size = (int(cutoff_size[0] * 1.00), int(cutoff_size[1] * 1.00))
                compose_kwargs = {
                    "prompt": prompt,
                    "target_height": prompt.height,
                    "target_width": prompt.width,
                    "cutoff": cutoff_size,
                    "dtype": dtype,
                }

                if prompt.init_image:
                    compose_kwargs.update(
                        {
                            "target_height": init_image.height,
                            "target_width": init_image.width,
                        }
                    )
                comp_image, comp_img_orig = _generate_composition_image(
                    **compose_kwargs, logging_context=lc
                )

                if comp_image is not None:
                    prompt.fix_faces = False  # done in composition
                    result_images["composition"] = comp_img_orig
                    result_images["composition-upscaled"] = comp_image
                    composition_strength = prompt.composition_strength
                    first_step = int((prompt.steps) * composition_strength)
                    noise_step = int((prompt.steps - 1) * composition_strength)
                    log_img(comp_img_orig, "comp_image")
                    log_img(comp_image, "comp_image_upscaled")
                    comp_image_t = pillow_img_to_torch_image(comp_image)
                    comp_image_t = comp_image_t.to(sd.lda.device, dtype=sd.lda.dtype)
                    init_latent = sd.lda.encode(comp_image_t)
                    compose_control_inputs: list[ControlInput]
                    if (
                        not prompt.compose_phase_controlnet
                        or prompt.model_weights.architecture.primary_alias
                        in (
                            "sdxl",
                            "sdxlinpaint",
                        )
                    ):
                        compose_control_inputs = []
                    else:
                        compose_control_inputs = [
                            ControlInput(mode="details", image=comp_image, strength=1),
                        ]
                    for control_input in compose_control_inputs:
                        (
                            controlnet,
                            control_image_t,
                            control_image_disp,
                        ) = prep_control_input(
                            control_input=control_input,
                            sd=sd,
                            init_image_t=None,
                            fit_width=prompt.width,
                            fit_height=prompt.height,
                        )
                        result_images[f"control-{control_input.mode}"] = (
                            control_image_disp
                        )
                        controlnets.append((controlnet, control_image_t))

                    _latent_h = init_latent.shape[2]
                    _latent_w = init_latent.shape[3]
                    if prompt.multidiffusion and (
                        _latent_h > MULTIDIFFUSION_TILE_SIZE
                        or _latent_w > MULTIDIFFUSION_TILE_SIZE
                    ):
                        with lc.timing("tile-captioning"):
                            tile_conditioning = _caption_tile_regions(
                                comp_image=comp_image,
                                tile_ys=_get_tile_positions(
                                    _latent_h,
                                    MULTIDIFFUSION_TILE_SIZE,
                                    MULTIDIFFUSION_STRIDE,
                                ),
                                tile_xs=_get_tile_positions(
                                    _latent_w,
                                    MULTIDIFFUSION_TILE_SIZE,
                                    MULTIDIFFUSION_STRIDE,
                                ),
                                tile_size=MULTIDIFFUSION_TILE_SIZE,
                                downsampling_factor=downsampling_factor,
                                global_prompts=prompt.prompts,
                                negative_prompts=prompt.negative_prompt,
                                sd=sd,
                            )

        if prompt.image_prompt:
            sd.set_image_prompt(
                prompt.image_prompt,
                scale=prompt.image_prompt_strength,
                model_type="plus",
            )
        for controlnet, control_image_t in controlnets:
            controlnet.set_controlnet_condition(
                control_image_t.to(device=sd.unet.device, dtype=sd.unet.dtype)
            )
            controlnet.inject()

        if prompt.solver_type.lower() == SolverName.DPMPP:
            sd.scheduler = DPMSolver(num_inference_steps=prompt.steps)
        elif prompt.solver_type.lower() == SolverName.DDIM:
            sd.scheduler = DDIM(num_inference_steps=prompt.steps)
        else:
            msg = f"Unknown solver type: {prompt.solver_type}"
            raise ValueError(msg)
        sd.scheduler.to(device=sd.unet.device, dtype=sd.unet.dtype)
        sd.set_inference_steps(prompt.steps, first_step=first_step)

        if hasattr(sd, "mask_latents") and mask_image is not None:
            masked_image = Image.composite(
                init_image, mask_image.convert("RGB"), mask_image
            )
            result_images["masked_image"] = masked_image
            sd.set_inpainting_conditions(
                target_image=init_image,
                mask=ImageOps.invert(mask_image),
                latents_size=shape[-2:],
            )
            sd.target_image_latents = sd.target_image_latents.to(
                dtype=sd.unet.dtype, device=sd.unet.device
            )
            sd.mask_latents = sd.mask_latents.to(
                dtype=sd.unet.dtype, device=sd.unet.device
            )
        elif prompt.inpaint_method == "patch" and mask_image is not None:
            import numpy as np
            import torch
            import torch.nn.functional as F

            from imaginairy.modules.fooocus_inpaint import (
                compute_inpaint_conditioning,
                inject_inpaint_features,
            )

            masked_image = Image.composite(
                init_image, mask_image.convert("RGB"), mask_image
            )
            result_images["masked_image"] = masked_image
            # mask_image here has white=keep after MaskMode inversion.
            # compute_inpaint_conditioning expects white=inpaint, so invert.
            inpaint_mask = ImageOps.invert(mask_image)
            with lc.timing("fooocus-conditioning"):
                features = compute_inpaint_conditioning(
                    sd=sd,
                    init_image=init_image,
                    mask_image=inpaint_mask,
                )
                inject_inpaint_features(sd.unet, features)

            # Prepare latent-space mask for per-step compositing (1=inpaint, 0=keep)
            mask_np = np.array(inpaint_mask).astype(np.float32) / 255.0
            _patch_latent_mask = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
            _patch_latent_mask = F.max_pool2d(
                _patch_latent_mask,
                kernel_size=downsampling_factor,
                stride=downsampling_factor,
            ).round()
            _patch_latent_mask = _patch_latent_mask.to(
                device=sd.unet.device, dtype=sd.unet.dtype
            )

        if init_latent is not None:
            noise_step = noise_step if noise_step is not None else first_step
            if first_step >= len(sd.scheduler.all_steps):
                noised_latent = init_latent
            else:
                noised_latent = sd.scheduler.add_noise(
                    x=init_latent, noise=noise, step=sd.scheduler.all_steps[noise_step]
                )

        with lc.timing("text-conditioning"):
            text_conditioning_kwargs = sd.calculate_text_conditioning_kwargs(
                positive_prompts=prompt.prompts,
                negative_prompts=prompt.negative_prompt,
                positive_conditioning_override=prompt.conditioning,
            )

            for k, v in text_conditioning_kwargs.items():
                text_conditioning_kwargs[k] = v.to(
                    device=sd.unet.device, dtype=sd.unet.dtype
                )
        x = noised_latent
        x = x.to(device=sd.unet.device, dtype=sd.unet.dtype)
        clear_gpu_cache()

        latent_h, latent_w = x.shape[2], x.shape[3]
        use_multidiffusion = prompt.multidiffusion and (
            latent_h > MULTIDIFFUSION_TILE_SIZE or latent_w > MULTIDIFFUSION_TILE_SIZE
        )

        with lc.timing("unet"):
            if use_multidiffusion:
                import torch

                tile_ys = _get_tile_positions(
                    latent_h, MULTIDIFFUSION_TILE_SIZE, MULTIDIFFUSION_STRIDE
                )
                tile_xs = _get_tile_positions(
                    latent_w, MULTIDIFFUSION_TILE_SIZE, MULTIDIFFUSION_STRIDE
                )
                logger.info(
                    f"MultiDiffusion: {len(tile_ys)}x{len(tile_xs)} tiles "
                    f"({len(tile_ys) * len(tile_xs)} per step) for {latent_w * 8}x{latent_h * 8} image"
                )
                # full-size controlnet conditions (already set on line 292)
                cn_full_conds = {id(cn): cond_t for cn, cond_t in controlnets}
                # full-size inpainting conditions (for sdxlinpaint model)
                _full_mask_latents = getattr(sd, "mask_latents", None)
                _full_target_latents = getattr(sd, "target_image_latents", None)
                px = downsampling_factor  # latent-to-pixel ratio
                for step in tqdm(
                    sd.steps, bar_format="    {l_bar}{bar}{r_bar}", leave=False
                ):
                    log_latent(x, "noisy_latent")
                    cumulative = torch.zeros_like(x)
                    num_updates = torch.zeros_like(x)
                    for ty in tile_ys:
                        for tx in tile_xs:
                            te, re = (
                                ty + MULTIDIFFUSION_TILE_SIZE,
                                tx + MULTIDIFFUSION_TILE_SIZE,
                            )
                            tile = x[:, :, ty:te, tx:re]
                            # crop each controlnet condition to matching pixel region
                            for cn, _ in controlnets:
                                cn.set_controlnet_condition(
                                    cn_full_conds[id(cn)][
                                        :, :, ty * px : te * px, tx * px : re * px
                                    ]
                                )
                            # crop inpainting conditions to matching tile region
                            if _full_mask_latents is not None:
                                sd.mask_latents = _full_mask_latents[:, :, ty:te, tx:re]
                            if _full_target_latents is not None:
                                sd.target_image_latents = _full_target_latents[
                                    :, :, ty:te, tx:re
                                ]
                            tile_kwargs = tile_conditioning.get(
                                (ty, tx), text_conditioning_kwargs
                            )
                            result = sd(
                                tile,
                                step=step,
                                condition_scale=prompt.prompt_strength,
                                **tile_kwargs,
                            )
                            cumulative[:, :, ty:te, tx:re] += result
                            num_updates[:, :, ty:te, tx:re] += 1
                    x = cumulative / num_updates
                    if lc.progress_latent_callback:
                        lc.progress_latent_callback(x)
                # restore full-size conditions
                for cn, _ in controlnets:
                    cn.set_controlnet_condition(cn_full_conds[id(cn)])
                if _full_mask_latents is not None:
                    sd.mask_latents = _full_mask_latents
                if _full_target_latents is not None:
                    sd.target_image_latents = _full_target_latents
            else:
                for step in tqdm(
                    sd.steps, bar_format="    {l_bar}{bar}{r_bar}", leave=False
                ):
                    log_latent(x, "noisy_latent")
                    if _patch_latent_mask is not None:
                        # Per-step latent masking: blend noised original into unmasked region
                        noised_orig = sd.scheduler.add_noise(
                            x=init_latent.to(x.device, x.dtype),
                            noise=noise,
                            step=step,
                        )
                        x = x * _patch_latent_mask + noised_orig * (
                            1 - _patch_latent_mask
                        )
                    x = sd(
                        x,
                        step=step,
                        condition_scale=prompt.prompt_strength,
                        **text_conditioning_kwargs,
                    )
                    if _patch_latent_mask is not None:
                        # Hard-set unmasked region to clean original after each step
                        x = x * _patch_latent_mask + init_latent.to(
                            x.device, x.dtype
                        ) * (1 - _patch_latent_mask)
                    if lc.progress_latent_callback:
                        lc.progress_latent_callback(x)
            # trying to clear memory. not sure if this helps
            sd.unet.set_context(context="self_attention_map", value={})
            sd.unet._reset_context()
            clear_gpu_cache()

        logger.debug("Decoding image")
        if x.device != sd.lda.device:
            sd.lda.to(x.device)
        clear_gpu_cache()
        with lc.timing("decode-img"):
            gen_img = sd.lda.decode_latents(x.to(dtype=sd.lda.dtype))

        if mask_image_orig and init_image:
            with lc.timing("combine-image"):
                result_images["pre-reconstitution"] = gen_img
                mask_final = mask_image_orig.copy()
                # mask_final = ImageOps.invert(mask_final)

                log_img(mask_final, "reconstituting mask")
                # gen_img = Image.composite(gen_img, init_image, mask_final)
                gen_img = combine_image(
                    original_img=init_image,
                    generated_img=gen_img,
                    mask_img=mask_final,
                )
                log_img(gen_img, "reconstituted image")

        upscaled_img = None
        rebuilt_orig_img = None

        if add_caption:
            with lc.timing("caption-img"):
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
                with lc.timing("face-enhancement"):
                    logger.info("Fixing 😊 's in 🖼  using CodeFormer...")
                    with lc.timing("face-enhancement"):
                        gen_img = enhance_faces(
                            gen_img, fidelity=prompt.fix_faces_fidelity
                        )
            if prompt.upscale:
                with lc.timing("upscaling"):
                    logger.info("Upscaling 🖼  using real-ESRGAN...")
                    with lc.timing("upscaling"):
                        upscaled_img = upscale_image(gen_img)

            # put the newly generated patch back into the original, full-size image
            if prompt.mask_modify_original and mask_image_orig and starting_image:
                logger.info("Combining inpainting with original image...")
                img_to_add_back_to_original = upscaled_img if upscaled_img else gen_img
                rebuilt_orig_img = combine_image(
                    original_img=starting_image,
                    generated_img=img_to_add_back_to_original,
                    mask_img=mask_image_orig,
                )

            if prompt.caption_text:
                caption_text = prompt.caption_text.format(prompt=prompt.prompt_text)
                add_caption_to_image(gen_img, caption_text)

        # todo: do something smarter
        result_images.update(
            {
                "upscaled": upscaled_img,
                "modified_original": rebuilt_orig_img,
                "mask_binary": mask_image_orig,
                "mask_grayscale": mask_grayscale,
            }
        )

        result = ImagineResult(
            img=gen_img,
            prompt=prompt,
            is_nsfw=safety_score.is_nsfw,
            safety_score=safety_score,
            result_images=result_images,
            performance_stats=lc.get_performance_stats(),
            progress_latents=progress_latents,
        )

        _most_recent_result = result
        _image_name = f"{image_name} " if image_name else ""
        logger.info(f"Generated {_image_name}image in {result.total_time():.1f}s")

        if result.performance_stats:
            log = logger.info if output_perf else logger.debug
            log(f"   Timings: {result.timings_str()}")
            if torch.cuda.is_available():
                log(f"   Peak VRAM: {result.gpu_str('memory_peak')}")
                log(f"   Peak VRAM Delta: {result.gpu_str('memory_peak_delta')}")
                log(f"   Ending VRAM: {result.gpu_str('memory_end')}")
        for controlnet, _ in controlnets:
            controlnet.eject()
        clear_gpu_cache()
        return result


def prep_control_input(
    control_input: ControlInput, sd, init_image_t, fit_width, fit_height
):
    from PIL import ImageOps

    from imaginairy.utils import get_device
    from imaginairy.utils.img_utils import (
        pillow_fit_image_within,
        pillow_img_to_torch_image,
    )
    from imaginairy.utils.log_utils import (
        log_img,
    )

    if control_input.image_raw is not None:
        control_image = control_input.image_raw
    elif control_input.image is not None:
        control_image = control_input.image
    else:
        raise ValueError("No control image found")
    assert control_image is not None
    control_image = control_image.convert("RGB")
    log_img(control_image, "control_image_input")
    control_image_input = pillow_fit_image_within(
        control_image,
        max_height=fit_height,
        max_width=fit_width,
    )
    if control_input.mode == "inpaint":
        control_image_input = ImageOps.invert(control_image_input)

    control_image_input_t = pillow_img_to_torch_image(control_image_input)
    control_image_input_t = control_image_input_t.to(get_device())

    if control_input.image_raw is None:
        from imaginairy.img_processors.control_modes import CONTROL_MODES

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

    log_img(control_image_disp, "control_image")

    if len(control_image_t.shape) == 3:
        raise ValueError("Control image must be 4D")

    if control_image_t.shape[1] != 3:
        raise ValueError("Control image must have 3 channels")

    if (
        control_input.mode != "inpaint" and control_image_t.min() < 0
    ) or control_image_t.max() > 1:
        msg = f"Control image must be in [0, 1] but we received {control_image_t.min()} and {control_image_t.max()}"
        raise ValueError(msg)

    if control_image_t.max() == control_image_t.min():
        msg = f"No control signal found in control image {control_input.mode}."
        raise ValueError(msg)

    control_config = CONTROL_CONFIG_SHORTCUTS.get(control_input.mode, None)
    if not control_config:
        msg = f"Unknown control mode: {control_input.mode}"
        raise ValueError(msg)
    from imaginairy.vendored.refiners.foundationals.latent_diffusion import (
        SD1ControlnetAdapter,
    )

    controlnet = SD1ControlnetAdapter(  # type: ignore
        name=control_input.mode,
        target=sd.unet,
        weights_location=control_config.weights_location,
    )

    controlnet.set_scale(control_input.strength)
    control_image_t = control_image_t.to(device=sd.unet.device, dtype=sd.unet.dtype)
    return controlnet, control_image_t, control_image_disp


def _generate_composition_image(
    prompt,
    target_height,
    target_width,
    cutoff: tuple[int, int] = (512, 512),
    dtype=None,
    logging_context=None,
):
    from PIL import Image

    from imaginairy.utils import default, get_default_dtype
    from imaginairy.utils.img_utils import calc_scale_to_fit_within
    from imaginairy.utils.named_resolutions import normalize_image_size

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
            "caption_text": None,
        },
    )

    result = generate_single_image(
        composition_prompt,
        dtype=dtype,
        logging_context=logging_context,
        output_perf=False,
        image_name="composition",
    )
    img = result.images["generated"]
    while img.width < target_width:
        from imaginairy.enhancers.upscale import upscale_image

        if prompt.fix_faces:
            from imaginairy.enhancers.face_restoration_codeformer import enhance_faces

            with logging_context.timing("face-enhancement"):
                logger.info("Fixing 😊 's in 🖼  using CodeFormer...")
                img = enhance_faces(img, fidelity=prompt.fix_faces_fidelity)
        with logging_context.timing("upscaling"):
            img = upscale_image(img, upscaler_model="ultrasharp")

    img = img.resize(
        (target_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )

    return img, result.images["generated"]

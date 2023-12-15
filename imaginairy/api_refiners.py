"""Functions for generating refined images"""

import logging
from typing import List, Optional

from imaginairy.config import CONTROL_CONFIG_SHORTCUTS
from imaginairy.schema import ControlInput, ImaginePrompt, MaskMode, WeightedPrompt

logger = logging.getLogger(__name__)


def _generate_single_image(
    prompt: ImaginePrompt,
    debug_img_callback=None,
    progress_img_callback=None,
    progress_img_interval_steps=3,
    progress_img_interval_min_s=0.1,
    add_caption=False,
    return_latent=False,
    dtype=None,
    half_mode=None,
):
    import torch.nn
    from PIL import Image, ImageOps
    from pytorch_lightning import seed_everything
    from refiners.foundationals.latent_diffusion.schedulers import DDIM, DPMSolver
    from tqdm import tqdm

    from imaginairy.api import (
        IMAGINAIRY_SAFETY_MODE,
        _generate_composition_image,
        combine_image,
    )
    from imaginairy.enhancers.clip_masking import get_img_mask
    from imaginairy.enhancers.describe_image_blip import generate_caption
    from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
    from imaginairy.enhancers.upscale_realesrgan import upscale_image
    from imaginairy.img_utils import (
        add_caption_to_image,
        pillow_fit_image_within,
        pillow_img_to_torch_image,
        pillow_mask_to_latent_mask,
    )
    from imaginairy.log_utils import (
        ImageLoggingContext,
        log_img,
        log_latent,
    )
    from imaginairy.model_manager import (
        get_diffusion_model_refiners,
        get_model_default_image_size,
    )
    from imaginairy.outpaint import outpaint_arg_str_parse, prepare_image_for_outpaint
    from imaginairy.safety import create_safety_score
    from imaginairy.samplers import SolverName
    from imaginairy.schema import ImagineResult
    from imaginairy.utils import get_device, randn_seeded

    if dtype is None:
        dtype = torch.float16 if half_mode else torch.float32

    get_device()
    clear_gpu_cache()
    prompt = prompt.make_concrete_copy()

    sd = get_diffusion_model_refiners(
        weights_config=prompt.model_weights,
        for_inpainting=prompt.should_use_inpainting
        and prompt.inpaint_method == "finetune",
        dtype=dtype,
    )

    seed_everything(prompt.seed)
    downsampling_factor = 8
    latent_channels = 4
    batch_size = 1

    mask_image = None
    mask_image_orig = None

    def latent_logger(latents):
        progress_latents.append(latents)

    with ImageLoggingContext(
        prompt=prompt,
        model=sd,
        debug_img_callback=debug_img_callback,
        progress_img_callback=progress_img_callback,
        progress_img_interval_steps=progress_img_interval_steps,
        progress_img_interval_min_s=progress_img_interval_min_s,
        progress_latent_callback=latent_logger
        if prompt.collect_progress_latents
        else None,
    ) as lc:
        sd.set_tile_mode(prompt.tile_mode)

        clip_text_embedding = _calc_conditioning(
            positive_prompts=prompt.prompts,
            negative_prompts=prompt.negative_prompt,
            positive_conditioning=prompt.conditioning,
            text_encoder=sd.clip_text_encoder,
        )
        clip_text_embedding = clip_text_embedding.to(device=sd.device, dtype=sd.dtype)

        result_images: dict[str, torch.Tensor | None | Image.Image] = {}
        progress_latents: list[torch.Tensor] = []
        first_step = 0
        mask_grayscale = None

        shape = [
            batch_size,
            latent_channels,
            prompt.height // downsampling_factor,
            prompt.width // downsampling_factor,
        ]

        init_latent = None
        noise_step = None

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
            init_image_t = init_image_t.to(device=sd.device, dtype=sd.dtype)
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
            get_device(), dtype=sd.dtype
        )
        noised_latent = noise
        controlnets = []

        if control_modes:
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
                if control_input.mode == "inpaint":
                    control_image_input = ImageOps.invert(control_image_input)

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
                    raise ValueError("Control image must be 4D")

                if control_image_t.shape[1] != 3:
                    raise ValueError("Control image must have 3 channels")

                if (
                    control_input.mode != "inpaint"
                    and control_image_t.min() < 0
                    or control_image_t.max() > 1
                ):
                    msg = f"Control image must be in [0, 1] but we received {control_image_t.min()} and {control_image_t.max()}"
                    raise ValueError(msg)

                if control_image_t.max() == control_image_t.min():
                    msg = f"No control signal found in control image {control_input.mode}."
                    raise ValueError(msg)

                control_config = CONTROL_CONFIG_SHORTCUTS.get(control_input.mode, None)
                if not control_config:
                    msg = f"Unknown control mode: {control_input.mode}"
                    raise ValueError(msg)
                from refiners.foundationals.latent_diffusion import SD1ControlnetAdapter

                controlnet = SD1ControlnetAdapter(  # type: ignore
                    name=control_input.mode,
                    target=sd.unet,  # type: ignore
                    weights_location=control_config.weights_location,
                )
                controlnet.set_scale(control_input.strength)

                controlnets.append((controlnet, control_image_t))

        if prompt.allow_compose_phase:
            cutoff_size = get_model_default_image_size(prompt.model_architecture)
            cutoff_size = (int(cutoff_size[0] * 1.30), int(cutoff_size[1] * 1.30))
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
            comp_image, comp_img_orig = _generate_composition_image(**compose_kwargs)

            if comp_image is not None:
                result_images["composition"] = comp_img_orig
                result_images["composition-upscaled"] = comp_image
                comp_cutoff = 0.50
                first_step = int((prompt.steps) * comp_cutoff)
                noise_step = int((prompt.steps - 1) * comp_cutoff)
                log_img(comp_img_orig, "comp_image")
                log_img(comp_image, "comp_image_upscaled")
                comp_image_t = pillow_img_to_torch_image(comp_image)
                comp_image_t = comp_image_t.to(sd.device, dtype=sd.dtype)
                init_latent = sd.lda.encode(comp_image_t)

        for controlnet, control_image_t in controlnets:
            controlnet.set_controlnet_condition(
                control_image_t.to(device=sd.device, dtype=sd.dtype)
            )
            controlnet.inject()
        if prompt.solver_type.lower() == SolverName.DPMPP:
            sd.scheduler = DPMSolver(num_inference_steps=prompt.steps)
        elif prompt.solver_type.lower() == SolverName.DDIM:
            sd.scheduler = DDIM(num_inference_steps=prompt.steps)
        else:
            msg = f"Unknown solver type: {prompt.solver_type}"
            raise ValueError(msg)
        sd.scheduler.to(device=sd.device, dtype=sd.dtype)
        sd.set_num_inference_steps(prompt.steps)

        if hasattr(sd, "mask_latents") and mask_image is not None:
            sd.set_inpainting_conditions(
                target_image=init_image,
                mask=ImageOps.invert(mask_image),
                latents_size=shape[-2:],
            )

        if init_latent is not None:
            noise_step = noise_step if noise_step is not None else first_step
            if first_step >= len(sd.steps):
                noised_latent = init_latent
            else:
                noised_latent = sd.scheduler.add_noise(
                    x=init_latent, noise=noise, step=sd.steps[noise_step]
                )

        x = noised_latent
        x = x.to(device=sd.device, dtype=sd.dtype)

        # if "cuda" in str(sd.lda.device):
        #     sd.lda.to("cpu")
        clear_gpu_cache()
        # print(f"moving unet to {sd.device}")
        # sd.unet.to(device=sd.device, dtype=sd.dtype)
        for step in tqdm(sd.steps[first_step:], bar_format="    {l_bar}{bar}{r_bar}"):
            log_latent(x, "noisy_latent")
            x = sd(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                condition_scale=prompt.prompt_strength,
            )

        clear_gpu_cache()

        logger.debug("Decoding image")
        if x.device != sd.lda.device:
            sd.lda.to(x.device)
        clear_gpu_cache()
        gen_img = sd.lda.decode_latents(x)

        if mask_image_orig and init_image:
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
            timings=lc.get_timings(),
            progress_latents=[],  # todo
        )

        _most_recent_result = result
        if result.timings:
            logger.info(f"Image Generated. Timings: {result.timings_str()}")
        for controlnet, _ in controlnets:
            controlnet.eject()
        clear_gpu_cache()
        return result


def _prompts_to_embeddings(prompts, text_encoder):
    import torch

    if not prompts:
        prompts = [WeightedPrompt(text="")]

    total_weight = sum(wp.weight for wp in prompts)
    if str(text_encoder.device) == "cpu":
        text_encoder = text_encoder.to(dtype=torch.float32)
    conditioning = sum(
        text_encoder(wp.text) * (wp.weight / total_weight) for wp in prompts
    )

    return conditioning


def _calc_conditioning(
    positive_prompts: Optional[List[WeightedPrompt]],
    negative_prompts: Optional[List[WeightedPrompt]],
    positive_conditioning,
    text_encoder,
):
    import torch

    from imaginairy.log_utils import log_conditioning

    # need to expand if doing batches
    neutral_conditioning = _prompts_to_embeddings(negative_prompts, text_encoder)
    log_conditioning(neutral_conditioning, "neutral conditioning")

    if positive_conditioning is None:
        positive_conditioning = _prompts_to_embeddings(positive_prompts, text_encoder)
    log_conditioning(positive_conditioning, "positive conditioning")

    clip_text_embedding = torch.cat(
        tensors=(neutral_conditioning, positive_conditioning), dim=0
    )
    return clip_text_embedding


def clear_gpu_cache():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

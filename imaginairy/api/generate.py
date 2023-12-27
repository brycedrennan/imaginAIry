"""Functions for generating and processing images"""

import logging
import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from imaginairy.schema import ImaginePrompt

logger = logging.getLogger(__name__)

# leave undocumented. I'd ask that no one publicize this flag. Just want a
# slight barrier to entry. Please don't use this is any way that's gonna cause
# the media or politicians to freak out about AI...
IMAGINAIRY_SAFETY_MODE = os.getenv("IMAGINAIRY_SAFETY_MODE", "strict")
if IMAGINAIRY_SAFETY_MODE in {"disabled", "classify"}:
    IMAGINAIRY_SAFETY_MODE = "relaxed"
elif IMAGINAIRY_SAFETY_MODE == "filter":
    IMAGINAIRY_SAFETY_MODE = "strict"

# we put this in the global scope so it can be used in the interactive shell
_most_recent_result = None


def imagine_image_files(
    prompts: "list[ImaginePrompt] | ImaginePrompt",
    outdir: str,
    precision: str = "autocast",
    record_step_images: bool = False,
    output_file_extension: str = "jpg",
    print_caption: bool = False,
    make_gif: bool = False,
    make_compare_gif: bool = False,
    return_filename_type: str = "generated",
    videogen: bool = False,
):
    from PIL import ImageDraw

    from imaginairy.api.video_sample import generate_video
    from imaginairy.utils import get_next_filenumber, prompt_normalized
    from imaginairy.utils.animations import make_bounce_animation
    from imaginairy.utils.img_utils import pillow_fit_image_within

    generated_imgs_path = os.path.join(outdir, "generated")
    os.makedirs(generated_imgs_path, exist_ok=True)

    base_count = get_next_filenumber(generated_imgs_path)
    output_file_extension = output_file_extension.lower()
    if output_file_extension not in {"jpg", "png"}:
        raise ValueError("Must output a png or jpg")

    if not isinstance(prompts, list):
        prompts = [prompts]

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
            f"{base_count:06}_{prompt.seed}_{prompt.solver_type.replace('_', '')}{prompt.steps}_"
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
                if videogen:
                    try:
                        generate_video(
                            input_path=filepath,
                        )
                    except FileNotFoundError as e:
                        logger.error(str(e))
                        exit(1)

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
    prompts: "list[ImaginePrompt] | str | ImaginePrompt",
    precision: str = "autocast",
    debug_img_callback: Callable | None = None,
    progress_img_callback: Callable | None = None,
    progress_img_interval_steps: int = 3,
    progress_img_interval_min_s=0.1,
    half_mode=None,
    add_caption: bool = False,
    unsafe_retry_count: int = 1,
):
    import torch.nn

    from imaginairy.api.generate_refiners import generate_single_image
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
        logger.warning("Running in CPU mode. It's gonna be slooooooow.")
        from imaginairy.utils.torch_installer import torch_version_check

        torch_version_check()

    if half_mode is None:
        half_mode = "cuda" in get_device() or get_device() == "mps"

    with torch.no_grad(), platform_appropriate_autocast(
        precision
    ), fix_torch_nn_layer_norm(), fix_torch_group_norm():
        for i, prompt in enumerate(prompts):
            concrete_prompt = prompt.make_concrete_copy()
            logger.info(
                f"🖼  Generating  {i + 1}/{num_prompts}: {concrete_prompt.prompt_description()}"
            )
            for attempt in range(unsafe_retry_count + 1):
                if attempt > 0 and isinstance(concrete_prompt.seed, int):
                    concrete_prompt.seed += 100_000_000 + attempt
                result = generate_single_image(
                    concrete_prompt,
                    debug_img_callback=debug_img_callback,
                    progress_img_callback=progress_img_callback,
                    progress_img_interval_steps=progress_img_interval_steps,
                    progress_img_interval_min_s=progress_img_interval_min_s,
                    half_mode=half_mode,
                    add_caption=add_caption,
                    dtype=torch.float16 if half_mode else torch.float32,
                )
                if not result.safety_score.is_filtered:
                    break
                if attempt < unsafe_retry_count:
                    logger.info("    Image was unsafe, retrying with new seed...")

            yield result

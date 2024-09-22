"""Functions for generating and processing images"""

import logging
import os
from pathlib import Path
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
    """
    Generates and saves image files based on given prompts, with options for animations and videos.

    Args:
        prompts (list[ImaginePrompt] | ImaginePrompt): A prompt or list of prompts for image generation.
        outdir (str): Directory path where the generated images and other files will be saved.
        precision (str, optional): Precision mode for image generation, defaults to 'autocast'.
        record_step_images (bool, optional): If True, saves step-by-step images of the generation process, defaults to False.
        output_file_extension (str, optional): File extension for output images, must be 'jpg' or 'png', defaults to 'jpg'.
        print_caption (bool, optional): If True, prints captions on the generated images, defaults to False.
        make_gif (bool, optional): If True, creates a GIF from the generation steps, defaults to False.
        make_compare_gif (bool, optional): If True, creates a comparison GIF with initial and generated images, defaults to False.
        return_filename_type (str, optional): Type of filenames to return, defaults to 'generated'.
        videogen (bool, optional): If True, generates a video from the generated images, defaults to False.

    Returns:
        list[str]: A list of filenames of the generated images.

    Raises:
        ValueError: If the output file extension is not 'jpg' or 'png'.
    """
    from PIL import ImageDraw

    from imaginairy.utils import get_next_filenumber, prompt_normalized

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
        primary_filename = save_image_result(
            result,
            base_count,
            outdir=outdir,
            output_file_extension=output_file_extension,
            primary_filename_type=return_filename_type,
            make_gif=make_gif,
            make_compare_gif=make_compare_gif,
        )
        if not primary_filename:
            continue
        result_filenames.append(primary_filename)
        if primary_filename and videogen:
            from imaginairy.api.video_sample import generate_video

            try:
                generate_video(
                    input_path=primary_filename,
                )
            except FileNotFoundError as e:
                logger.error(str(e))
                exit(1)

        base_count += 1
        del result

    return result_filenames


def save_image_result(
    result,
    base_count: int,
    outdir: str | Path,
    output_file_extension: str,
    primary_filename_type,
    make_gif=False,
    make_compare_gif=False,
):
    from imaginairy.utils import prompt_normalized
    from imaginairy.utils.animations import make_bounce_animation
    from imaginairy.utils.img_utils import pillow_fit_image_within

    prompt = result.prompt
    if prompt.is_intermediate:
        # we don't save intermediate images
        return

    img_str = ""
    if prompt.init_image:
        img_str = f"_img2img-{prompt.init_image_strength}"

    basefilename = (
        f"{base_count:06}_{prompt.seed}_{prompt.solver_type.replace('_', '')}{prompt.steps}_"
        f"PS{prompt.prompt_strength}{img_str}_{prompt_normalized(prompt.prompt_text)}"
    )
    primary_filename = None
    for image_type in result.images:
        subpath = os.path.join(outdir, image_type)
        os.makedirs(subpath, exist_ok=True)
        filepath = os.path.join(
            subpath, f"{basefilename}_[{image_type}].{output_file_extension}"
        )
        result.save(filepath, image_type=image_type)
        logger.info(f"        {image_type:<22} {filepath}")

        if image_type == primary_filename_type:
            primary_filename = filepath

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
        image_type = "gif"
        logger.info(f"        {image_type:<22} {filepath}")
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
        image_type = "gif"
        logger.info(f"        {image_type:<22} {filepath}")

    return primary_filename


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
    """
    Generates images based on the provided prompts using the ImaginAIry API.

    Args:
        prompts (list[ImaginePrompt] | str | ImaginePrompt): A prompt or list of prompts for image generation.
            Can be a string, a single ImaginePrompt instance, or a list of ImaginePrompt instances.
        precision (str, optional): The precision mode for image generation, defaults to 'autocast'.
        debug_img_callback (Callable, optional): Callback function for debugging images, defaults to None.
        progress_img_callback (Callable, optional): Callback function called at intervals with progress images, defaults to None.
        progress_img_interval_steps (int, optional): Number of steps between each progress image callback, defaults to 3.
        progress_img_interval_min_s (float, optional): Minimum seconds between each progress image callback, defaults to 0.1.
        half_mode: If set, determines whether to use half precision mode for image generation, defaults to None.
        add_caption (bool, optional): Flag to add captions to the generated images, defaults to False.
        unsafe_retry_count (int, optional): Number of retries for generating an image if it is deemed unsafe, defaults to 1.

    Yields:
        The generated image(s) based on the provided prompts.
    """
    import torch.nn

    from imaginairy.schema import ImaginePrompt
    from imaginairy.utils import (
        check_torch_version,
        fix_torch_group_norm,
        fix_torch_nn_layer_norm,
        get_device,
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

    with torch.no_grad(), fix_torch_nn_layer_norm(), fix_torch_group_norm():
        for i, prompt in enumerate(prompts):
            concrete_prompt = prompt.make_concrete_copy()
            prog_text = f"{i + 1}/{num_prompts}"
            logger.info(f"🖼  {prog_text} {concrete_prompt.prompt_description()}")
            # Determine which generate function to use based on the model
            if (
                concrete_prompt.model_architecture
                and concrete_prompt.model_architecture.name.lower() == "flux"
            ):
                from imaginairy.api.generate_flux import (
                    generate_single_image as generate_single_flux_image,
                )

                generate_func = generate_single_flux_image
            else:
                from imaginairy.api.generate_refiners import (
                    generate_single_image as generate_single_image_refiners,
                )

                generate_func = generate_single_image_refiners

            for attempt in range(unsafe_retry_count + 1):
                if attempt > 0 and isinstance(concrete_prompt.seed, int):
                    concrete_prompt.seed += 100_000_000 + attempt
                result = generate_func(
                    concrete_prompt,
                    debug_img_callback=debug_img_callback,
                    progress_img_callback=progress_img_callback,
                    progress_img_interval_steps=progress_img_interval_steps,
                    progress_img_interval_min_s=progress_img_interval_min_s,
                    add_caption=add_caption,
                    dtype=torch.float16 if half_mode else torch.float32,
                    output_perf=True,
                )
                if not result.safety_score or not result.safety_score.is_filtered:
                    break
                if attempt < unsafe_retry_count:
                    logger.info("    Image was unsafe, retrying with new seed...")

            yield result

import logging
import math

import click

from imaginairy import config

logger = logging.getLogger(__name__)


def _imagine_cmd(
    ctx,
    prompt_texts,
    negative_prompt,
    prompt_strength,
    init_image,
    init_image_strength,
    outdir,
    output_file_extension,
    repeats,
    height,
    width,
    steps,
    seed,
    upscale,
    fix_faces,
    fix_faces_fidelity,
    sampler_type,
    log_level,
    quiet,
    show_work,
    tile,
    tile_x,
    tile_y,
    allow_compose_phase,
    mask_image,
    mask_prompt,
    mask_mode,
    mask_modify_original,
    outpaint,
    caption,
    precision,
    model_weights_path,
    model_config_path,
    prompt_library_path,
    version=False,
    make_gif=False,
    make_compare_gif=False,
    arg_schedules=None,
    make_compilation_animation=False,
    caption_text="",
    control_inputs=None,
):
    """Have the AI generate images. alias:imagine."""

    if ctx.invoked_subcommand is not None:
        return

    if version:
        from imaginairy.version import get_version

        print(get_version())
        return

    if quiet:
        log_level = "ERROR"

    import sys

    if len(sys.argv) > 1:
        msg = (
            "✨ Generate images faster using a persistent shell session. Just run `aimg` to start. "
            "This makes generation and editing much quicker since the model can stay loaded in memory.\n"
        )
        print(msg)

    from imaginairy.log_utils import configure_logging

    configure_logging(log_level)

    init_images = [init_image] if isinstance(init_image, str) else init_image

    from imaginairy.utils import glob_expand_paths

    num_prexpaned_init_images = len(init_images)
    init_images = glob_expand_paths(init_images)

    if len(init_images) < num_prexpaned_init_images:
        msg = f"Could not find any images matching the glob pattern(s) {init_image}. Are you sure the file(s) exists?"
        raise ValueError(msg)

    total_image_count = len(prompt_texts) * max(len(init_images), 1) * repeats
    logger.info(
        f"Received {len(prompt_texts)} prompt(s) and {len(init_images)} input image(s). Will repeat the generations {repeats} times to create {total_image_count} images."
    )

    from imaginairy import ImaginePrompt, LazyLoadingImage, imagine_image_files

    new_init_images = []
    for _init_image in init_images:
        if _init_image and _init_image.startswith("http"):
            _init_image = LazyLoadingImage(url=_init_image)
        else:
            _init_image = LazyLoadingImage(filepath=_init_image)
        new_init_images.append(_init_image)
    init_images = new_init_images
    if not init_images:
        init_images = [None]

    if mask_image:
        if mask_image.startswith("http"):
            mask_image = LazyLoadingImage(url=mask_image)
        else:
            mask_image = LazyLoadingImage(filepath=mask_image)

    prompts = []
    prompt_expanding_iterators = {}
    from imaginairy.enhancers.prompt_expansion import expand_prompts

    for _ in range(repeats):
        for prompt_text in prompt_texts:
            if prompt_text not in prompt_expanding_iterators:
                prompt_expanding_iterators[prompt_text] = expand_prompts(
                    n=math.inf,
                    prompt_text=prompt_text,
                    prompt_library_paths=prompt_library_path,
                )
            prompt_iterator = prompt_expanding_iterators[prompt_text]
            if tile:
                _tile_mode = "xy"
            elif tile_x:
                _tile_mode = "x"
            elif tile_y:
                _tile_mode = "y"
            else:
                _tile_mode = ""

            for _init_image in init_images:
                prompt = ImaginePrompt(
                    prompt=next(prompt_iterator),
                    negative_prompt=negative_prompt,
                    prompt_strength=prompt_strength,
                    init_image=_init_image,
                    init_image_strength=init_image_strength,
                    control_inputs=control_inputs,
                    seed=seed,
                    sampler_type=sampler_type,
                    steps=steps,
                    height=height,
                    width=width,
                    mask_image=mask_image,
                    mask_prompt=mask_prompt,
                    mask_mode=mask_mode,
                    mask_modify_original=mask_modify_original,
                    outpaint=outpaint,
                    upscale=upscale,
                    fix_faces=fix_faces,
                    fix_faces_fidelity=fix_faces_fidelity,
                    tile_mode=_tile_mode,
                    allow_compose_phase=allow_compose_phase,
                    model=model_weights_path,
                    model_config_path=model_config_path,
                    caption_text=caption_text,
                )
                from imaginairy.prompt_schedules import (
                    parse_schedule_strs,
                    prompt_mutator,
                )

                if arg_schedules:
                    schedules = parse_schedule_strs(arg_schedules)
                    for new_prompt in prompt_mutator(prompt, schedules):
                        prompts.append(new_prompt)
                else:
                    prompts.append(prompt)

    filenames = imagine_image_files(
        prompts,
        outdir=outdir,
        record_step_images=show_work,
        output_file_extension=output_file_extension,
        print_caption=caption,
        precision=precision,
        make_gif=make_gif,
        make_compare_gif=make_compare_gif,
    )
    if make_compilation_animation:
        import os.path

        ext = make_compilation_animation

        compilation_outdir = os.path.join(outdir, "compilations")
        os.makedirs(compilation_outdir, exist_ok=True)
        base_count = len(os.listdir(compilation_outdir))
        new_filename = os.path.join(
            compilation_outdir, f"{base_count:04d}_compilation.{ext}"
        )
        comp_imgs = [LazyLoadingImage(filepath=f) for f in filenames]
        comp_imgs.reverse()

        from imaginairy.animations import make_slideshow_animation

        make_slideshow_animation(
            outpath=new_filename,
            imgs=comp_imgs,
            image_pause_ms=1000,
        )

        logger.info(f"[compilation] saved to: {new_filename}")


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def replace_option(options, option_name, new_option):
    for i, option in enumerate(options):
        if option.name == option_name:
            options[i] = new_option
            return
    msg = f"Option {option_name} not found"
    raise ValueError(msg)


def remove_option(options, option_name):
    for i, option_dec in enumerate(options):

        def temp_f():
            return True

        temp_f = option_dec(temp_f)
        option = temp_f.__click_params__[0]

        if option.name == option_name:
            del options[i]
            return
    msg = f"Option {option_name} not found"
    raise ValueError(msg)


common_options = [
    click.option(
        "--negative-prompt",
        default=None,
        show_default=False,
        help="Negative prompt. Things to try and exclude from images. Same negative prompt will be used for all images.",
    ),
    click.option(
        "--prompt-strength",
        default=7.5,
        show_default=True,
        help="How closely to follow the prompt. Image looks unnatural at higher values",
    ),
    click.option(
        "--init-image",
        metavar="PATH|URL",
        help="Starting image.",
        multiple=True,
    ),
    click.option(
        "--init-image-strength",
        default=None,
        show_default=False,
        type=float,
        help="Starting image strength. Between 0 and 1.",
    ),
    click.option(
        "--outdir",
        default="./outputs",
        show_default=True,
        type=click.Path(),
        help="Where to write results to.",
    ),
    click.option(
        "--output-file-extension",
        default="jpg",
        show_default=True,
        type=click.Choice(["jpg", "png"]),
        help="Where to write results to.",
    ),
    click.option(
        "-r",
        "--repeats",
        default=1,
        show_default=True,
        type=int,
        help="How many times to repeat the renders. If you provide two prompts and --repeat=3 then six images will be generated.",
    ),
    click.option(
        "-h",
        "--height",
        default=None,
        show_default=True,
        type=int,
        help="Image height. Should be multiple of 8.",
    ),
    click.option(
        "-w",
        "--width",
        default=None,
        show_default=True,
        type=int,
        help="Image width. Should be multiple of 8.",
    ),
    click.option(
        "--steps",
        default=None,
        type=int,
        show_default=True,
        help="How many diffusion steps to run. More steps, more detail, but with diminishing returns.",
    ),
    click.option(
        "--seed",
        default=None,
        type=int,
        help="What seed to use for randomness. Allows reproducible image renders.",
    ),
    click.option("--upscale", is_flag=True),
    click.option("--fix-faces", is_flag=True),
    click.option(
        "--fix-faces-fidelity",
        default=None,
        type=float,
        help="How faithful to the original should face enhancement be. 1 = best fidelity, 0 = best looking face.",
    ),
    click.option(
        "--sampler-type",
        "--sampler",
        default=config.DEFAULT_SAMPLER,
        show_default=True,
        type=click.Choice(config.SAMPLER_TYPE_OPTIONS),
        help="What sampling strategy to use.",
    ),
    click.option(
        "--log-level",
        default="INFO",
        show_default=True,
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
        help="What level of logs to show.",
    ),
    click.option(
        "--quiet",
        "-q",
        is_flag=True,
        help="Suppress logs. Alias of `--log-level ERROR`.",
    ),
    click.option(
        "--show-work",
        default=False,
        is_flag=True,
        help="Output a debug images to `steps` folder.",
    ),
    click.option(
        "--tile",
        is_flag=True,
        help="Any images rendered will be tileable in both X and Y directions.",
    ),
    click.option(
        "--tile-x",
        is_flag=True,
        help="Any images rendered will be tileable in the X direction.",
    ),
    click.option(
        "--tile-y",
        is_flag=True,
        help="Any images rendered will be tileable in the Y direction.",
    ),
    click.option(
        "--allow-compose-phase/--no-compose-phase",
        default=True,
        help="Allow the image to be composed at a lower resolution.",
    ),
    click.option(
        "--mask-image",
        metavar="PATH|URL",
        help="A mask to use for inpainting. White gets painted, Black is left alone.",
    ),
    click.option(
        "--mask-prompt",
        help=(
            "Describe what you want masked and the AI will mask it for you. "
            "You can describe complex masks with AND, OR, NOT keywords and parentheses. "
            "The strength of each mask can be modified with {*1.5} notation. \n\n"
            "Examples:  \n"
            "car AND (wheels{*1.1} OR trunk OR engine OR windows OR headlights) AND NOT (truck OR headlights){*10}\n"
            "fruit|fruit stem"
        ),
    ),
    click.option(
        "--mask-mode",
        default="replace",
        show_default=True,
        type=click.Choice(["keep", "replace"]),
        help="Should we replace the masked area or keep it?",
    ),
    click.option(
        "--mask-modify-original",
        default=True,
        is_flag=True,
        help="After the inpainting is done, apply the changes to a copy of the original image.",
    ),
    click.option(
        "--outpaint",
        help=(
            "Specify in what directions to expand the image. Values will be snapped such that output image size is multiples of 8. Examples\n"
            "  `--outpaint up10,down300,left50,right50`\n"
            "  `--outpaint u10,d300,l50,r50`\n"
            "  `--outpaint all200`\n"
            "  `--outpaint a200`\n"
        ),
        default="",
    ),
    click.option(
        "--caption",
        default=False,
        is_flag=True,
        help="Generate a text description of the generated image.",
    ),
    click.option(
        "--precision",
        help="Evaluate at this precision.",
        type=click.Choice(["full", "autocast"]),
        default="autocast",
        show_default=True,
    ),
    click.option(
        "--model-weights-path",
        "--model",
        help=f"Model to use. Should be one of {', '.join(config.MODEL_SHORT_NAMES)}, or a path to custom weights.",
        show_default=True,
        default=config.DEFAULT_MODEL,
    ),
    click.option(
        "--model-config-path",
        help="Model config file to use. If a model name is specified, the appropriate config will be used.",
        show_default=True,
        default=None,
    ),
    click.option(
        "--prompt-library-path",
        help="Path to folder containing phrase lists in txt files. Use txt filename in prompt: {_filename_}.",
        type=click.Path(exists=True),
        default=None,
        multiple=True,
    ),
    click.option(
        "--version",
        default=False,
        is_flag=True,
        help="Print the version and exit.",
    ),
    click.option(
        "--gif",
        "make_gif",
        default=False,
        is_flag=True,
        help="Create a gif of the generation.",
    ),
    click.option(
        "--compare-gif",
        "make_compare_gif",
        default=False,
        is_flag=True,
        help="Create a gif comparing the original image to the modified one.",
    ),
    click.option(
        "--arg-schedule",
        "arg_schedules",
        multiple=True,
        help="Schedule how an argument should change over several generations. Format: `--arg-schedule arg_name[start:end:increment]` or `--arg-schedule arg_name[val,val2,val3]`",
    ),
    click.option(
        "--compilation-anim",
        "make_compilation_animation",
        default=None,
        type=click.Choice(["gif", "mp4"]),
        help="Generate an animation composed of all the images generated in this run.  Defaults to gif but `--compilation-anim mp4` will generate an mp4 instead.",
    ),
    click.option(
        "--caption-text",
        "caption_text",
        default=None,
        help="Specify the text to write onto the image",
        type=str,
    ),
]

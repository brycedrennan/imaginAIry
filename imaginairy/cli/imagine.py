import click

from imaginairy.cli.shared import (
    ImagineColorsCommand,
    _imagine_cmd,
    add_options,
    common_options,
)


@click.command(
    context_settings={"max_content_width": 140},
    cls=ImagineColorsCommand,
    name="imagine",
)
@click.argument("prompt_texts", nargs=-1)
@add_options(common_options)
@click.option(
    "--control-image",
    metavar="PATH|URL",
    help=(
        "Image used for control signal in image generation. "
        "For example if control-mode is depth, then the generated image will match the depth map "
        "extracted from the control image. "
        "Defaults to the `--init-image`"
    ),
    multiple=False,
)
@click.option(
    "--control-image-raw",
    metavar="PATH|URL",
    help=(
        "Preprocessed image used for control signal in image generation. Like `--control-image` but "
        " expects the already extracted signal.  For example the raw control image would be a depth map or"
        "pose information."
    ),
    multiple=False,
)
@click.option(
    "--control-mode",
    default=None,
    show_default=False,
    type=click.Choice(
        [
            "",
            "canny",
            "depth",
            "normal",
            "hed",
            "openpose",
            "shuffle",
            "edit",
            "inpaint",
            "details",
        ]
    ),
    help="how the control image is used as signal",
)
@click.pass_context
def imagine_cmd(
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
    version,  # noqa
    make_gif,
    make_compare_gif,
    arg_schedules,
    make_compilation_animation,
    caption_text,
    control_image,
    control_image_raw,
    control_mode,
):
    """
    Generate images via AI.

    Can be invoked via either `aimg imagine` or just `imagine`.
    """
    return _imagine_cmd(
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
        version,  # noqa
        make_gif,
        make_compare_gif,
        arg_schedules,
        make_compilation_animation,
        caption_text,
        control_image,
        control_image_raw,
        control_mode,
    )

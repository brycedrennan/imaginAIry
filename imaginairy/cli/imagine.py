"""Command-line interface for AI-driven image generation"""

import click

from imaginairy.cli.clickshell_mod import ImagineColorsCommand
from imaginairy.cli.shared import (
    _imagine_cmd,
    add_options,
    common_options,
    imaginairy_click_context,
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
    multiple=True,
)
@click.option(
    "--control-image-raw",
    metavar="PATH|URL",
    help=(
        "Preprocessed image used for control signal in image generation. Like `--control-image` but "
        " expects the already extracted signal.  For example the raw control image would be a depth map or"
        "pose information."
    ),
    multiple=True,
)
@click.option(
    "--control-strength",
    help=("Strength of the control signal."),
    multiple=True,
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
            "details",
            "normal",
            "hed",
            "openpose",
            "shuffle",
            "edit",
            "inpaint",
            "colorize",
            "qrcode",
        ]
    ),
    help="how the control image is used as signal",
    multiple=True,
)
@click.option(
    "--videogen",
    is_flag=True,
    default=False,
    show_default=False,
    help="Turns the generated photo into video",
)
@click.pass_context
@imaginairy_click_context()
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
    size,
    steps,
    seed,
    upscale,
    fix_faces,
    fix_faces_fidelity,
    solver,
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
    composition_strength,
    precision,
    model_weights_path,
    model_architecture,
    prompt_library_path,
    version,
    make_gif,
    make_compare_gif,
    arg_schedules,
    make_compilation_animation,
    caption_text,
    control_image,
    control_image_raw,
    control_strength,
    control_mode,
    videogen,
):
    """
    Generate images via AI.

    Can be invoked via either `aimg imagine` or just `imagine`.
    """
    from imaginairy.schema import ControlInput, LazyLoadingImage
    from imaginairy.utils import named_resolutions
    from imaginairy.utils.text_image import image_from_textimg_str

    # hacky method of getting order of control images (mixing raw and normal images)
    control_images = [
        (o, path)
        for o, path in ImagineColorsCommand._option_order
        if o.name in ("control_image", "control_image_raw")
    ]
    control_strengths = [
        strength
        for o, strength in ImagineColorsCommand._option_order
        if o.name == "control_strength"
    ]

    control_inputs = []
    if control_mode:
        for i, cm in enumerate(control_mode):
            option = index_default(control_images, i, None)
            control_strength = index_default(control_strengths, i, 1.0)

            if option is None:
                control_image = None
                control_image_raw = None
            elif option[0].name == "control_image":
                control_image = option[1]
                control_image_raw = None
                if control_image:
                    if control_image.startswith("http"):
                        control_image = LazyLoadingImage(url=control_image)
                    elif control_image.startswith("textimg="):
                        (
                            resolved_width,
                            resolved_height,
                        ) = named_resolutions.normalize_image_size(
                            size if size else 512
                        )

                        control_image = image_from_textimg_str(
                            control_image, resolved_width, resolved_height
                        )
            else:
                control_image = None
                control_image_raw = option[1]
                if control_image_raw and control_image_raw.startswith("http"):
                    control_image_raw = LazyLoadingImage(url=control_image_raw)
            control_inputs.append(
                ControlInput(
                    image=control_image,
                    image_raw=control_image_raw,
                    strength=float(control_strength),
                    mode=cm,
                )
            )

    return _imagine_cmd(
        ctx=ctx,
        prompt_texts=prompt_texts,
        negative_prompt=negative_prompt,
        prompt_strength=prompt_strength,
        init_image=init_image,
        init_image_strength=init_image_strength,
        outdir=outdir,
        output_file_extension=output_file_extension,
        repeats=repeats,
        size=size,
        steps=steps,
        seed=seed,
        upscale=upscale,
        fix_faces=fix_faces,
        fix_faces_fidelity=fix_faces_fidelity,
        solver=solver,
        log_level=log_level,
        quiet=quiet,
        show_work=show_work,
        tile=tile,
        tile_x=tile_x,
        tile_y=tile_y,
        allow_compose_phase=allow_compose_phase,
        mask_image=mask_image,
        mask_prompt=mask_prompt,
        mask_mode=mask_mode,
        mask_modify_original=mask_modify_original,
        outpaint=outpaint,
        caption=caption,
        composition_strength=composition_strength,
        precision=precision,
        model_weights_path=model_weights_path,
        model_architecture=model_architecture,
        prompt_library_path=prompt_library_path,
        version=version,
        make_gif=make_gif,
        make_compare_gif=make_compare_gif,
        arg_schedules=arg_schedules,
        make_compilation_animation=make_compilation_animation,
        caption_text=caption_text,
        control_inputs=control_inputs,
        videogen=videogen,
    )


def index_default(items, index, default):
    try:
        return items[index]
    except IndexError:
        return default


if __name__ == "__main__":
    imagine_cmd()

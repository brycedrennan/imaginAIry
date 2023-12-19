"""Command for AI-powered image editing"""

import click

from imaginairy import config
from imaginairy.cli.shared import (
    _imagine_cmd,
    add_options,
    common_options,
    remove_option,
)

edit_options = common_options.copy()


remove_option(edit_options, "model_weights_path")
remove_option(edit_options, "init_image")
remove_option(edit_options, "init_image_strength")
remove_option(edit_options, "negative_prompt")
remove_option(edit_options, "allow_compose_phase")


@click.command("edit")
@click.argument("image_paths", metavar="PATH|URL", required=True, nargs=-1)
@click.option(
    "--image-strength",
    default=0.1,
    show_default=False,
    type=float,
    help="Starting image strength. Between 0 and 1.",
)
@click.option("--prompt", "-p", required=True, multiple=True)
@click.option(
    "--model-weights-path",
    "--model",
    help=f"Model to use. Should be one of {', '.join(config.IMAGE_WEIGHTS_SHORT_NAMES)}, or a path to custom weights.",
    show_default=True,
    default="sd15",
)
@click.option(
    "--negative-prompt",
    default=None,
    show_default=False,
    help="Negative prompt. Things to try and exclude from images. Same negative prompt will be used for all images. A default negative prompt is used if none is selected.",
)
@add_options(edit_options)
@click.pass_context
def edit_cmd(
    ctx,
    image_paths,
    image_strength,
    prompt,
    negative_prompt,
    prompt_strength,
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
    mask_image,
    mask_prompt,
    mask_mode,
    mask_modify_original,
    outpaint,
    caption,
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
    composition_strength,
):
    """
    Edit an image via AI.

    Provide paths or URLs to images and directions on how to alter them.

    Example: aimg edit --prompt "make the dog red" my-dog.jpg my-dog2.jpg

    Same as calling `aimg imagine --model edit --init-image my-dog.jpg --init-image-strength 1` except this command
    can batch edit images.
    """
    from imaginairy.schema import ControlInput

    allow_compose_phase = False
    control_inputs = [ControlInput(image=None, image_raw=None, mode="edit", strength=1)]

    return _imagine_cmd(
        ctx=ctx,
        prompt_texts=prompt,
        negative_prompt=negative_prompt,
        prompt_strength=prompt_strength,
        init_image=image_paths,
        init_image_strength=image_strength,
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
        composition_strength=composition_strength,
        control_inputs=control_inputs,
    )

import logging
import math

import click
from click_help_colors import HelpColorsCommand, HelpColorsMixin
from click_shell import Shell

from imaginairy import config

logger = logging.getLogger(__name__)

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
]


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
    raise ValueError(f"Option {option_name} not found")


def remove_option(options, option_name):
    for i, option_dec in enumerate(options):

        def temp_f():
            return True

        temp_f = option_dec(temp_f)
        option = temp_f.__click_params__[0]

        if option.name == option_name:
            del options[i]
            return
    raise ValueError(f"Option {option_name} not found")


class ColorShell(HelpColorsMixin, Shell):
    pass


class ImagineColorsCommand(HelpColorsCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.help_headers_color = "yellow"
        self.help_options_color = "green"


@click.command(
    prompt="🤖🧠> ",
    intro="Starting imaginAIry shell...",
    help_headers_color="yellow",
    help_options_color="green",
    context_settings={"max_content_width": 140},
    cls=ColorShell,
)
@click.pass_context
def aimg(ctx):
    """
    🤖🧠 ImaginAIry.

    Pythonic generation of images via AI
    """
    import sys

    is_shell = len(sys.argv) == 1
    if is_shell:
        print(ctx.get_help())


aimg.command_class = ImagineColorsCommand


@click.command(context_settings={"max_content_width": 140}, cls=ImagineColorsCommand)
@click.argument("prompt_texts", nargs=-1)
@add_options(common_options)
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
    )


@aimg.command("edit-demo")
@click.argument("image_paths", metavar="PATH|URL", required=True, nargs=-1)
@click.option(
    "--outdir",
    default="./outputs",
    show_default=True,
    type=click.Path(),
    help="Where to write results to.",
)
@click.option(
    "-h",
    "--height",
    default=512,
    show_default=True,
    type=int,
    help="Image height. Should be multiple of 8.",
)
@click.option(
    "-w",
    "--width",
    default=512,
    show_default=True,
    type=int,
    help="Image width. Should be multiple of 8.",
)
def edit_demo(image_paths, outdir, height, width):
    """Make some fun pre-set edits to input photos."""

    from imaginairy.log_utils import configure_logging
    from imaginairy.surprise_me import create_surprise_me_images

    configure_logging()
    for image_path in image_paths:
        create_surprise_me_images(
            image_path, outdir=outdir, make_gif=True, width=width, height=height
        )


edit_options = common_options.copy()
remove_option(edit_options, "model_weights_path")
remove_option(edit_options, "init_image")
remove_option(edit_options, "init_image_strength")
remove_option(edit_options, "negative_prompt")
remove_option(edit_options, "allow_compose_phase")


@aimg.command("edit")
@click.argument("image_paths", metavar="PATH|URL", required=True, nargs=-1)
@click.option(
    "--image-strength",
    default=1,
    show_default=False,
    type=float,
    help="Starting image strength. Between 0 and 1.",
)
@click.option("--prompt", "-p", required=True, multiple=True)
@click.option(
    "--model-weights-path",
    "--model",
    help=f"Model to use. Should be one of {', '.join(config.MODEL_SHORT_NAMES)}, or a path to custom weights.",
    show_default=True,
    default="edit",
)
@click.option(
    "--negative-prompt",
    default=None,
    show_default=False,
    help="Negative prompt. Things to try and exclude from images. Same negative prompt will be used for all images. A default negative prompt is used if none is selected.",
)
@add_options(edit_options)
@click.pass_context
def edit_image(  # noqa
    ctx,
    image_paths,
    image_strength,
    prompt,
    negative_prompt,
    prompt_strength,
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
):
    """
    Edit an image via AI.

    Provide paths or URLs to images and directions on how to alter them.

    Example: aimg edit --prompt "make the dog red" my-dog.jpg my-dog2.jpg

    Same as calling `aimg imagine --model edit --init-image my-dog.jpg --init-image-strength 1` except this command
    can batch edit images.
    """
    allow_compose_phase = False
    return _imagine_cmd(
        ctx,
        prompt,
        negative_prompt,
        prompt_strength,
        image_paths,
        image_strength,
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
    )


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
    version=False,  # noqa
    make_gif=False,
    make_compare_gif=False,
    arg_schedules=None,
    make_compilation_animation=False,
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

    if isinstance(init_image, str):
        init_images = [init_image]
    else:
        init_images = init_image
    total_image_count = len(prompt_texts) * max(len(init_images), 1) * repeats
    logger.info(
        f"Received {len(prompt_texts)} prompt(s) and {len(init_images)} input image(s). Will repeat the generations {repeats} times to create {total_image_count} images."
    )

    from imaginairy import ImaginePrompt, LazyLoadingImage, imagine_image_files

    new_init_images = []
    for _init_image in init_images:
        if _init_image and _init_image.startswith("http"):
            _init_image = LazyLoadingImage(url=_init_image)
        new_init_images.append(_init_image)
    init_images = new_init_images
    if not init_images:
        init_images = [None]

    if mask_image and mask_image.startswith("http"):
        mask_image = LazyLoadingImage(url=mask_image)

    if init_image_strength is None:
        if outpaint or mask_image or mask_prompt:
            init_image_strength = 0
        else:
            init_image_strength = 0.6

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
                    next(prompt_iterator),
                    negative_prompt=negative_prompt,
                    prompt_strength=prompt_strength,
                    init_image=_init_image,
                    init_image_strength=init_image_strength,
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

        from imaginairy.animations import make_bounce_animation

        make_bounce_animation(
            outpath=new_filename,
            imgs=comp_imgs,
            start_pause_duration_ms=1500,
            end_pause_duration_ms=1000,
        )

        logger.info(f"[compilation] saved to: {new_filename}")


@aimg.command()
def version():
    """Print the version."""
    from imaginairy.version import get_version

    print(get_version())


@click.argument("image_filepaths", nargs=-1)
@click.option(
    "--outdir",
    default="./outputs/upscaled",
    show_default=True,
    type=click.Path(),
    help="Where to write results to.",
)
@aimg.command("upscale")
def upscale_cmd(image_filepaths, outdir):
    """
    Upscale an image 4x using AI.
    """
    import os.path

    from tqdm import tqdm

    from imaginairy import LazyLoadingImage
    from imaginairy.enhancers.upscale_realesrgan import upscale_image

    os.makedirs(outdir, exist_ok=True)

    for p in tqdm(image_filepaths):
        savepath = os.path.join(outdir, os.path.basename(p))
        if p.startswith("http"):
            img = LazyLoadingImage(url=p)
        else:
            img = LazyLoadingImage(filepath=p)
        logger.info(
            f"Upscaling {p} from {img.width}x{img.height} to {img.width * 4}x{img.height * 4} and saving it to {savepath}"
        )

        img = upscale_image(img)

        img.save(os.path.join(outdir, os.path.basename(p)))


@click.argument("image_filepaths", nargs=-1)
@aimg.command()
def describe(image_filepaths):
    """Generate text descriptions of images."""

    from imaginairy import LazyLoadingImage
    from imaginairy.enhancers.describe_image_blip import generate_caption

    imgs = []
    for p in image_filepaths:
        if p.startswith("http"):
            img = LazyLoadingImage(url=p)
        else:
            img = LazyLoadingImage(filepath=p)
        imgs.append(img)
    for img in imgs:
        print(generate_caption(img.copy()))


@click.option(
    "--concept-label",
    help=(
        'The concept you are training on. Usually "a photo of [person or thing] [classname]" is what you should use.'
    ),
    required=True,
)
@click.option(
    "--concept-images-dir",
    type=click.Path(),
    required=True,
    help="Where to find the pre-processed concept images to train on.",
)
@click.option(
    "--class-label",
    help=(
        'What class of things does the concept belong to. For example, if you are training on "a painting of a George Washington", '
        'you might use "a painting of a man" as the class label. We use this to prevent the model from overfitting.'
    ),
    default="a photo of *",
)
@click.option(
    "--class-images-dir",
    type=click.Path(),
    required=True,
    help="Where to find the pre-processed class images to train on.",
)
@click.option(
    "--n-class-images",
    type=int,
    default=300,
    help="Number of class images to generate.",
)
@click.option(
    "--model-weights-path",
    "--model",
    "model",
    help=f"Model to use. Should be one of {', '.join(config.MODEL_SHORT_NAMES)}, or a path to custom weights.",
    show_default=True,
    default=config.DEFAULT_MODEL,
)
@click.option(
    "--person",
    "is_person",
    is_flag=True,
    help="Set if images are of a person. Will use face detection and enhancement.",
)
@click.option(
    "-y",
    "preconfirmed",
    is_flag=True,
    default=False,
    help="Bypass input confirmations.",
)
@click.option(
    "--skip-prep",
    is_flag=True,
    default=False,
    help="Skip the image preparation step.",
)
@click.option(
    "--skip-class-img-gen",
    is_flag=True,
    default=False,
    help="Skip the class image generation step.",
)
@aimg.command()
def train_concept(
    concept_label,
    concept_images_dir,
    class_label,
    class_images_dir,
    n_class_images,
    model,
    is_person,
    preconfirmed,
    skip_prep,
    skip_class_img_gen,
):
    """
    Teach the model a new concept (a person, thing, style, etc).

    Provided a directory of concept images, a concept token, and a class token, this command will train the model
    to generate images of that concept.

    \b
    This happens in a 3-step process:
      1. Cropping and resizing your training images. If --person is set we crop to include the face.
      2. Generating a set of class images to train on.  This helps prevent overfitting.
      3. Training the model on the concept and class images.

    The output of this command is a new model weights file that you can use with the --model option.

    \b
    ## Instructions
     1. Gather a set of images of the concept you want to train on. They should show the subject from a variety of angles
     and in a variety of situations.
     2. Train the model.
     - Concept label: For a person, firstnamelastname should be fine.
        - If all the training images are photos you should add "a photo of" to the beginning of the concept label.
     - Class label: This is the category of the things beings trained on.  For people this is typically "person", "man"
     or "woman".
        - If all the training images are photos you should add "a photo of" to the beginning of the class label.
        - CLass images will be generated for you if you do not provide them.
    3. Stop training before it overfits. I haven't figured this out yet.


    For example, if you were training on photos of a man named bill hamilton you could run the following:

    \b
    aimg train-concept \\
        --person \\
        --concept-label "photo of billhamilton man" \\
        --concept-images-dir ./images/billhamilton \\
        --class-label "photo of a man" \\
        --class-images-dir ./images/man

    When you use the model you should prompt with `firstnamelastname classname` (e.g. `billhamilton man`).

    You can find a lot of relevant instructions here: https://github.com/JoePenna/Dreambooth-Stable-Diffusion
    """
    from imaginairy.utils import get_device

    if "mps" in get_device():
        click.secho(
            "⚠️  MPS (MacOS) is not supported for training. Please use a GPU or CPU.",
            fg="yellow",
        )
        return

    import os.path

    from imaginairy.train import train_diffusion_model
    from imaginairy.training_tools.image_prep import (
        create_class_images,
        get_image_filenames,
        prep_images,
    )

    target_size = 512
    # Step 1. Crop and enhance the training images
    prepped_images_path = os.path.join(concept_images_dir, "prepped-images")
    image_filenames = get_image_filenames(concept_images_dir)
    click.secho(
        f'\n🤖🧠 Training "{concept_label}" based on {len(image_filenames)} images.\n'
    )

    if not skip_prep:
        msg = (
            f"Creating cropped copies of the {len(image_filenames)} concept images\n"
            f"    Is Person: {is_person}\n"
            f"    Source: {concept_images_dir}\n"
            f"    Dest: {prepped_images_path}\n"
        )
        logger.info(msg)
        if not is_person:
            click.secho("⚠️  the --person flag was not set. ", fg="yellow")

        if not preconfirmed and not click.confirm("Continue?"):
            return

        prep_images(
            images_dir=concept_images_dir, is_person=is_person, target_size=target_size
        )
        concept_images_dir = prepped_images_path

    if not skip_class_img_gen:
        # Step 2. Generate class images
        class_image_filenames = get_image_filenames(class_images_dir)
        images_needed = max(n_class_images - len(class_image_filenames), 0)
        logger.info(f"Generating {n_class_images} class images in {class_images_dir}")
        logger.info(
            f"{len(class_image_filenames)} existing class images found so only generating {images_needed}."
        )
        if not preconfirmed and not click.confirm("Continue?"):
            return
        create_class_images(
            class_description=class_label,
            output_folder=class_images_dir,
            num_images=n_class_images,
        )

    logger.info("Training the model...")
    if not preconfirmed and not click.confirm("Continue?"):
        return

    # Step 3. Train the model
    train_diffusion_model(
        concept_label=concept_label,
        concept_images_dir=concept_images_dir,
        class_label=class_label,
        class_images_dir=class_images_dir,
        weights_location=model,
        logdir="logs",
        learning_rate=1e-6,
        accumulate_grad_batches=32,
    )


@click.argument(
    "images_dir",
    required=True,
)
@click.option(
    "--person",
    "is_person",
    is_flag=True,
    help="Set if images are of a person. Will use face detection and enhancement.",
)
@click.option(
    "--target-size",
    default=512,
    type=int,
    show_default=True,
)
@aimg.command("prep-images")
def prepare_images(images_dir, is_person, target_size):
    """
    Prepare a folder of images for training.

    Prepped images will be written to the `prepped-images` subfolder.

    All images will be cropped and resized to (default) 512x512.
    Upscaling and face enhancement will be applied as needed to smaller images.

    Examples:
        aimg prep-images --person ./images/selfies
        aimg prep-images ./images/toy-train
    """

    from imaginairy.training_tools.image_prep import prep_images

    prep_images(images_dir=images_dir, is_person=is_person, target_size=target_size)


@click.argument("ckpt_paths", nargs=-1)
@aimg.command("prune-ckpt")
def prune_ckpt(ckpt_paths):
    """
    Prune a checkpoint file.

    This will remove the optimizer state from the checkpoint file.
    This is useful if you want to use the checkpoint file for inference and save a lot of disk space

    Example:
        aimg prune-ckpt ./path/to/checkpoint.ckpt
    """
    from imaginairy.training_tools.prune_model import prune_diffusion_ckpt

    click.secho("Pruning checkpoint files...")
    for p in ckpt_paths:
        prune_diffusion_ckpt(p)


@aimg.command("system-info")
def system_info():
    """
    Display system information. Submit this when reporting bugs.
    """
    from imaginairy.debug_info import get_debug_info

    for k, v in get_debug_info().items():
        k += ":"
        click.secho(f"{k: <30} {v}")


aimg.add_command(imagine_cmd, name="imagine")

if __name__ == "__main__":
    imagine_cmd()  # noqa
    # from cProfile import Profile
    # from pyprof2calltree import convert, visualize
    # profiler = Profile()
    # profiler.runctx("imagine_cmd.main(standalone_mode=False)", locals(), globals())
    # convert(profiler.getstats(), 'imagine.kgrind')
    # visualize(profiler.getstats())

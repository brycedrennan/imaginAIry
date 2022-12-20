import logging
import math

import click
from click_shell import shell

from imaginairy import LazyLoadingImage, config, generate_caption
from imaginairy.api import imagine_image_files
from imaginairy.config import MODEL_SHORT_NAMES
from imaginairy.enhancers.prompt_expansion import expand_prompts
from imaginairy.log_utils import configure_logging
from imaginairy.samplers import SAMPLER_TYPE_OPTIONS
from imaginairy.schema import ImaginePrompt

logger = logging.getLogger(__name__)


@click.command()
@click.argument("prompt_texts", nargs=-1)
@click.option(
    "--negative-prompt",
    default=config.DEFAULT_NEGATIVE_PROMPT,
    show_default=True,
    help="Negative prompt. Things to try and exclude from images. Same negative prompt will be used for all images.",
)
@click.option(
    "--prompt-strength",
    default=7.5,
    show_default=True,
    help="How closely to follow the prompt. Image looks unnatural at higher values",
)
@click.option(
    "--init-image",
    metavar="PATH|URL",
    help="Starting image.",
)
@click.option(
    "--init-image-strength",
    default=0.6,
    show_default=True,
    help="Starting image strength. Between 0 and 1.",
)
@click.option(
    "--outdir",
    default="./outputs",
    show_default=True,
    type=click.Path(),
    help="Where to write results to.",
)
@click.option(
    "-r",
    "--repeats",
    default=1,
    show_default=True,
    type=int,
    help="How many times to repeat the renders. If you provide two prompts and --repeat=3 then six images will be generated.",
)
@click.option(
    "-h",
    "--height",
    default=None,
    show_default=True,
    type=int,
    help="Image height. Should be multiple of 64.",
)
@click.option(
    "-w",
    "--width",
    default=None,
    show_default=True,
    type=int,
    help="Image width. Should be multiple of 64.",
)
@click.option(
    "--steps",
    default=None,
    type=int,
    show_default=True,
    help="How many diffusion steps to run. More steps, more detail, but with diminishing returns.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="What seed to use for randomness. Allows reproducible image renders.",
)
@click.option("--upscale", is_flag=True)
@click.option("--fix-faces", is_flag=True)
@click.option(
    "--fix-faces-fidelity",
    default=None,
    type=float,
    help="How faithful to the original should face enhancement be. 1 = best fidelity, 0 = best looking face.",
)
@click.option(
    "--sampler-type",
    "--sampler",
    default=config.DEFAULT_SAMPLER,
    show_default=True,
    type=click.Choice(SAMPLER_TYPE_OPTIONS),
    help="What sampling strategy to use.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="What level of logs to show.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress logs. Alias of `--log-level ERROR`.",
)
@click.option(
    "--show-work",
    default=False,
    is_flag=True,
    help="Output a debug images to `steps` folder.",
)
@click.option(
    "--tile",
    is_flag=True,
    help="Any images rendered will be tileable in both X and Y directions.",
)
@click.option(
    "--tile-x",
    is_flag=True,
    help="Any images rendered will be tileable in the X direction.",
)
@click.option(
    "--tile-y",
    is_flag=True,
    help="Any images rendered will be tileable in the Y direction.",
)
@click.option(
    "--mask-image",
    metavar="PATH|URL",
    help="A mask to use for inpainting. White gets painted, Black is left alone.",
)
@click.option(
    "--mask-prompt",
    help=(
        "Describe what you want masked and the AI will mask it for you. "
        "You can describe complex masks with AND, OR, NOT keywords and parentheses. "
        "The strength of each mask can be modified with {*1.5} notation. \n\n"
        "Examples:  \n"
        "car AND (wheels{*1.1} OR trunk OR engine OR windows OR headlights) AND NOT (truck OR headlights){*10}\n"
        "fruit|fruit stem"
    ),
)
@click.option(
    "--mask-mode",
    default="replace",
    show_default=True,
    type=click.Choice(["keep", "replace"]),
    help="Should we replace the masked area or keep it?",
)
@click.option(
    "--mask-modify-original",
    default=True,
    is_flag=True,
    help="After the inpainting is done, apply the changes to a copy of the original image.",
)
@click.option(
    "--caption",
    default=False,
    is_flag=True,
    help="Generate a text description of the generated image.",
)
@click.option(
    "--precision",
    help="Evaluate at this precision.",
    type=click.Choice(["full", "autocast"]),
    default="autocast",
    show_default=True,
)
@click.option(
    "--model-weights-path",
    "--model",
    help=f"Model to use. Should be one of {', '.join(MODEL_SHORT_NAMES)}, or a path to custom weights.",
    show_default=True,
    default=config.DEFAULT_MODEL,
)
@click.option(
    "--prompt-library-path",
    help="Path to folder containing phrase lists in txt files. Use txt filename in prompt: {_filename_}.",
    type=click.Path(exists=True),
    default=None,
    multiple=True,
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
    caption,
    precision,
    model_weights_path,
    prompt_library_path,
):
    """Have the AI generate images. alias:imagine"""
    if ctx.invoked_subcommand is not None:
        return

    if quiet:
        log_level = "ERROR"
    configure_logging(log_level)

    total_image_count = len(prompt_texts) * repeats
    logger.info(
        f"🤖🧠 imaginAIry received {len(prompt_texts)} prompt(s) and will repeat them {repeats} times to create {total_image_count} images."
    )

    if init_image and init_image.startswith("http"):
        init_image = LazyLoadingImage(url=init_image)

    if mask_image and mask_image.startswith("http"):
        mask_image = LazyLoadingImage(url=mask_image)

    prompts = []
    prompt_expanding_iterators = {}
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

            prompt = ImaginePrompt(
                next(prompt_iterator),
                negative_prompt=negative_prompt,
                prompt_strength=prompt_strength,
                init_image=init_image,
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
                upscale=upscale,
                fix_faces=fix_faces,
                fix_faces_fidelity=fix_faces_fidelity,
                tile_mode=_tile_mode,
                model=model_weights_path,
            )
            prompts.append(prompt)

    imagine_image_files(
        prompts,
        outdir=outdir,
        record_step_images=show_work,
        output_file_extension="jpg",
        print_caption=caption,
        precision=precision,
    )


@shell(prompt="imaginAIry> ", intro="Starting imaginAIry...")
def aimg():
    pass


@click.argument("image_filepaths", nargs=-1)
@aimg.command()
def describe(image_filepaths):
    """Generate text descriptions of images"""
    imgs = []
    for p in image_filepaths:
        if p.startswith("http"):
            img = LazyLoadingImage(url=p)
        else:
            img = LazyLoadingImage(filepath=p)
        imgs.append(img)
    for img in imgs:
        print(generate_caption(img.copy()))


aimg.add_command(imagine_cmd, name="imagine")

if __name__ == "__main__":
    imagine_cmd()  # noqa
    # from cProfile import Profile
    # from pyprof2calltree import convert, visualize
    # profiler = Profile()
    # profiler.runctx("imagine_cmd.main(standalone_mode=False)", locals(), globals())
    # convert(profiler.getstats(), 'imagine.kgrind')
    # visualize(profiler.getstats())

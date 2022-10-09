import logging.config
import math

import click
from click_shell import shell

from imaginairy import LazyLoadingImage, generate_caption
from imaginairy.api import imagine_image_files
from imaginairy.enhancers.prompt_expansion import expand_prompts
from imaginairy.samplers.base import SAMPLER_TYPE_OPTIONS
from imaginairy.schema import ImaginePrompt
from imaginairy.suppress_logs import suppress_annoying_logs_and_warnings

logger = logging.getLogger(__name__)


def configure_logging(level="INFO"):
    fmt = "%(message)s"
    if level == "DEBUG":
        fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {"format": fmt},
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # Default is stderr
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["default"],
                "level": "WARNING",
                "propagate": False,
            },
            "imaginairy": {"handlers": ["default"], "level": level, "propagate": False},
            "transformers.modeling_utils": {
                "handlers": ["default"],
                "level": "ERROR",
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)


@click.command()
@click.argument("prompt_texts", nargs=-1)
@click.option(
    "--prompt-strength",
    default=7.5,
    show_default=True,
    help="How closely to follow the prompt. Image looks unnatural at higher values",
)
@click.option(
    "--init-image",
    help="Starting image. filepath or url",
)
@click.option(
    "--init-image-strength",
    default=0.6,
    show_default=True,
    help="Starting image.",
)
@click.option("--outdir", default="./outputs", help="where to write results to")
@click.option(
    "-r",
    "--repeats",
    default=1,
    type=int,
    help="How many times to repeat the renders. If you provide two prompts and --repeat=3 then six images will be generated",
)
@click.option(
    "-h",
    "--height",
    default=512,
    type=int,
    help="image height. should be multiple of 64",
)
@click.option(
    "-w", "--width", default=512, type=int, help="image width. should be multiple of 64"
)
@click.option(
    "--steps",
    default=40,
    type=int,
    show_default=True,
    help="How many diffusion steps to run. More steps, more detail, but with diminishing returns",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="What seed to use for randomness. Allows reproducible image renders",
)
@click.option("--upscale", is_flag=True)
@click.option("--fix-faces", is_flag=True)
@click.option(
    "--fix-faces-fidelity",
    default=None,
    help="How faithful to the original should face enhancement be. 1 = best fidelity, 0 = best looking face",
)
@click.option(
    "--sampler-type",
    default="plms",
    type=click.Choice(SAMPLER_TYPE_OPTIONS),
    help="What sampling strategy to use",
)
@click.option("--ddim-eta", default=0.0, type=float)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="What level of logs to show.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress logs. Alias of `--log-level ERROR`",
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
    help="Any images rendered will be tileable.",
)
@click.option(
    "--mask-image",
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
    type=click.Choice(["keep", "replace"]),
    help="Should we replace the masked area or keep it?",
)
@click.option(
    "--mask-modify-original",
    default=True,
    is_flag=True,
    help="After the inpainting is done, apply the changes to a copy of the original image",
)
@click.option(
    "--caption",
    default=False,
    is_flag=True,
    help="Generate a text description of the generated image",
)
@click.option(
    "--precision",
    help="evaluate at this precision",
    type=click.Choice(["full", "autocast"]),
    default="autocast",
)
@click.option(
    "--model-weights-path",
    help="path to model weights file. by default we use stable diffusion 1.4",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--prompt-library-path",
    help="path to folder containing phaselists in txt files. use txt filename in prompt: {_filename_}",
    type=click.Path(exists=True),
    default=None,
    multiple=True,
)
@click.pass_context
def imagine_cmd(
    ctx,
    prompt_texts,
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
    ddim_eta,
    log_level,
    quiet,
    show_work,
    tile,
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
    suppress_annoying_logs_and_warnings()
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
    if fix_faces_fidelity is not None:
        fix_faces_fidelity = float(fix_faces_fidelity)
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
            prompt = ImaginePrompt(
                next(prompt_iterator),
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
                tile_mode=tile,
            )
            prompts.append(prompt)

    imagine_image_files(
        prompts,
        outdir=outdir,
        ddim_eta=ddim_eta,
        record_step_images=show_work,
        output_file_extension="jpg",
        print_caption=caption,
        precision=precision,
        model_weights_path=model_weights_path,
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

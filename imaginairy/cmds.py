import logging.config

import click

from imaginairy.api import load_model

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
    help="Starting image.",
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
@click.option(
    "--upscale-method", default="realesrgan", type=click.Choice(["realesrgan"])
)
@click.option("--fix-faces", is_flag=True)
@click.option("--fix-faces-method", default="gfpgan", type=click.Choice(["gfpgan"]))
@click.option(
    "--sampler-type",
    default="PLMS",
    type=click.Choice(["PLMS", "DDIM"]),
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
    "--show-work",
    default=["none"],
    type=click.Choice(["none", "images", "video"]),
    multiple=True,
    help="Make a video showing the image being created",
)
@click.option(
    "--tile",
    is_flag=True,
    help="Any images rendered will be tileable.  Unfortunately cannot be controlled at the per-image level yet",
)
def imagine_cmd(
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
    upscale_method,
    fix_faces,
    fix_faces_method,
    sampler_type,
    ddim_eta,
    log_level,
    show_work,
    tile,
):
    """Render an image"""
    configure_logging(log_level)

    from imaginairy.api import imagine_image_files
    from imaginairy.schema import ImaginePrompt

    total_image_count = len(prompt_texts) * repeats
    logger.info(
        f"🤖🧠 imaginAIry received {len(prompt_texts)} prompt(s) and will repeat them {repeats} times to create {total_image_count} images."
    )
    if init_image and sampler_type != "DDIM":
        sampler_type = "DDIM"

    prompts = []
    load_model(tile_mode=tile)
    for _ in range(repeats):
        for prompt_text in prompt_texts:
            prompt = ImaginePrompt(
                prompt_text,
                prompt_strength=prompt_strength,
                init_image=init_image,
                init_image_strength=init_image_strength,
                seed=seed,
                sampler_type=sampler_type,
                steps=steps,
                height=height,
                width=width,
                upscale=upscale,
                fix_faces=fix_faces,
            )
            prompts.append(prompt)

    imagine_image_files(
        prompts,
        outdir=outdir,
        ddim_eta=ddim_eta,
        record_step_images="images" in show_work,
        tile_mode=tile,
        output_file_extension="png",
    )


if __name__ == "__main__":
    imagine_cmd()  # noqa

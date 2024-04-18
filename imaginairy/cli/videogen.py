"""Command for generating AI-powered videos"""

import logging

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--start-image",
    default="other/images/sound-music.jpg",
    help="Input path for image file.",
)
@click.option("--num-frames", default=None, type=int, help="Number of frames.")
@click.option(
    "-s", "--steps", default=None, type=int, help="Number of diffusion steps."
)
@click.option(
    "--model",
    default="svd",
    help="Model to use. One of: svd, svd-xt, svd-image-decoder, svd-xt-image-decoder",
)
@click.option(
    "--fps", default=6, type=int, help="FPS for the AI to target when generating video"
)
@click.option(
    "--size",
    default="1024,576",
    show_default=True,
    type=str,
    help="Video dimensions. Can be a named size, single integer, or WIDTHxHEIGHT pair. Should be multiple of 8. Examples: SVD, 512x512, 4k, UHD, 8k, 512, 1080p",
)
@click.option("--output-fps", default=None, type=int, help="FPS for the output video")
@click.option(
    "--output-format",
    default="webp",
    help="Output video format",
    type=click.Choice(["webp", "mp4", "gif"]),
)
@click.option(
    "--motion-amount",
    default=127,
    type=int,
    help="How much motion to generate. value between 0 and 255.",
)
@click.option(
    "-r",
    "--repeats",
    default=1,
    show_default=True,
    type=int,
    help="How many times to repeat the renders. ",
)
@click.option("--cond-aug", default=0.02, type=float, help="Conditional augmentation.")
@click.option(
    "--seed", default=None, type=int, help="Seed for random number generator."
)
@click.option(
    "--decoding_t", default=1, type=int, help="Number of frames decoded at a time."
)
@click.option("--output_folder", default=None, help="Output folder.")
def videogen_cmd(
    start_image,
    num_frames,
    steps,
    model,
    fps,
    size,
    output_fps,
    output_format,
    motion_amount,
    repeats,
    cond_aug,
    seed,
    decoding_t,
    output_folder,
):
    """
    AI generate a video from an image

    Example:

        aimg videogen --start-image assets/rocket-wide.png

    """
    from imaginairy.api.video_sample import generate_video
    from imaginairy.utils.log_utils import configure_logging

    configure_logging()

    output_fps = output_fps or fps
    try:
        generate_video(
            input_path=start_image,
            num_frames=num_frames,
            num_steps=steps,
            model_name=model,
            fps_id=fps,
            size=size,
            output_fps=output_fps,
            output_format=output_format,
            motion_bucket_id=motion_amount,
            cond_aug=cond_aug,
            seed=seed,
            decoding_t=decoding_t,
            output_folder=output_folder,
            repetitions=repeats,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)

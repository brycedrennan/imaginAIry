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
@click.option("--num-steps", default=None, type=int, help="Number of steps.")
@click.option(
    "--model",
    default="svd",
    help="Model to use. One of: svd, svd_xt, svd_image_decoder, svd_xt_image_decoder",
)
@click.option(
    "--fps", default=6, type=int, help="FPS for the AI to target when generating video"
)
@click.option("--output-fps", default=None, type=int, help="FPS for the output video")
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
    num_steps,
    model,
    fps,
    output_fps,
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
    from imaginairy.log_utils import configure_logging
    from imaginairy.video_sample import generate_video

    configure_logging()

    output_fps = output_fps or fps
    for i in range(repeats):
        logger.info(f"Generating video from image {start_image}")
        generate_video(
            input_path=start_image,
            num_frames=num_frames,
            num_steps=num_steps,
            model_name=model,
            fps_id=fps,
            output_fps=output_fps,
            motion_bucket_id=motion_amount,
            cond_aug=cond_aug,
            seed=seed,
            decoding_t=decoding_t,
            output_folder=output_folder,
        )

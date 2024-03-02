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
    help="Model to use. One of: svd, svd_xt, svd_image_decoder, svd_xt_image_decoder",
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
    import os
    from glob import glob

    from imaginairy.api.video_sample import generate_video
    from imaginairy.utils import default
    from imaginairy.utils.log_utils import configure_logging

    configure_logging()

    output_fps = output_fps or fps

    all_images = []

    try:
        all_images.extend(load_images(start_image))
    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)

    output_folder_str = default(output_folder, "outputs/video/")

    os.makedirs(output_folder_str, exist_ok=True)

    samples, output_fps = generate_video(
        input_images=all_images,
        num_frames=num_frames,
        num_steps=steps,
        model_name=model,
        fps_id=fps,
        size=size,
        output_fps=output_fps,
        motion_bucket_id=motion_amount,
        cond_aug=cond_aug,
        seed=seed,
        decoding_t=decoding_t,
        repetitions=repeats,
    )

    for sample in samples:
        base_count = len(glob(os.path.join(output_folder_str, "*.*"))) + 1
        source_slug = make_safe_filename(sample)
        video_filename = (
            f"{base_count:06d}_{model}_{seed}_{fps}fps_{source_slug}.{output_format}"
        )
        video_path = os.path.join(output_folder_str, video_filename)

        from imaginairy.api.video_sample import save_video_bounce

        save_video_bounce(samples, video_path, output_fps)


def load_images(start_image):
    from pathlib import Path

    from imaginairy.schema import LazyLoadingImage

    if start_image.startswith("http"):
        image = LazyLoadingImage(url=start_image).as_pillow()
        return [image]
    else:
        path = Path(start_image)
        if path.is_file():
            if any(start_image.endswith(x) for x in ["jpg", "jpeg", "png"]):
                return [LazyLoadingImage(filepath=start_image).as_pillow()]
            else:
                raise ValueError("Path is not a valid image file.")
        elif path.is_dir():
            all_img_paths = sorted(
                [
                    str(f)
                    for f in path.iterdir()
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]
            )
            if len(all_img_paths) == 0:
                raise ValueError("Folder does not contain any images.")
            return [
                LazyLoadingImage(filepath=image).as_pillow() for image in all_img_paths
            ]
        else:
            msg = f"Could not find file or folder at {start_image}"
            raise FileNotFoundError(msg)


def make_safe_filename(input_string):
    import os
    import re

    stripped_url = re.sub(r"^https?://[^/]+/", "", input_string)

    # Remove directory path if present
    base_name = os.path.basename(stripped_url)

    # Remove file extension
    name_without_extension = os.path.splitext(base_name)[0]

    # Keep only alphanumeric characters and dashes
    safe_name = re.sub(r"[^a-zA-Z0-9\-]", "", name_without_extension)

    return safe_name

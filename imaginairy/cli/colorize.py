import logging

import click

logger = logging.getLogger(__name__)


@click.argument("image_filepaths", nargs=-1)
@click.option(
    "--outdir",
    default="./outputs/colorized",
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
    "--caption",
    default="",
    show_default=False,
    help="Description of the photo. If not provided, it will be generated automatically.",
)
@click.command("colorize")
def colorize_cmd(image_filepaths, outdir, repeats, caption):
    """
    Colorize images using AI. Doesn't work very well yet.
    """
    import os.path

    from tqdm import tqdm

    from imaginairy import LazyLoadingImage
    from imaginairy.colorize import colorize_img
    from imaginairy.log_utils import configure_logging

    configure_logging()

    os.makedirs(outdir, exist_ok=True)
    base_count = len(os.listdir(outdir))
    for _ in range(repeats):
        for p in tqdm(image_filepaths):
            base_count += 1
            filename = f"{base_count:06d}_{os.path.basename(p)}".lower()
            savepath = os.path.join(outdir, filename)
            if p.startswith("http"):
                img = LazyLoadingImage(url=p)
            elif os.path.isdir(p):
                print(f"Skipping directory: {p}")
                continue
            else:
                img = LazyLoadingImage(filepath=p)
            logger.info(f"Colorizing {p} and saving it to {savepath}")

            img = colorize_img(img, caption=caption)

            img.save(savepath)

import logging

import click

logger = logging.getLogger(__name__)


@click.argument("image_filepaths", nargs=-1)
@click.option(
    "--outdir",
    default="./outputs/upscaled",
    show_default=True,
    type=click.Path(),
    help="Where to write results to.",
)
@click.command("upscale")
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

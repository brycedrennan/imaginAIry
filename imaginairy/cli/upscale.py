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
@click.option("--fix-faces", is_flag=True)
@click.option(
    "--fix-faces-fidelity",
    default=1,
    type=float,
    help="How faithful to the original should face enhancement be. 1 = best fidelity, 0 = best looking face.",
)
@click.command("upscale")
def upscale_cmd(image_filepaths, outdir, fix_faces, fix_faces_fidelity):
    """
    Upscale an image 4x using AI.
    """
    import os.path

    from tqdm import tqdm

    from imaginairy import LazyLoadingImage
    from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
    from imaginairy.enhancers.upscale_realesrgan import upscale_image
    from imaginairy.utils import glob_expand_paths

    os.makedirs(outdir, exist_ok=True)
    image_filepaths = glob_expand_paths(image_filepaths)
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
        if fix_faces:
            img = enhance_faces(img, fidelity=fix_faces_fidelity)

        img.save(os.path.join(outdir, os.path.basename(p)))

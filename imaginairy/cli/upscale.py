"""Command for upscaling images with AI"""

import logging
import os.path
from datetime import datetime, timezone

import click

from imaginairy.config import DEFAULT_UPSCALE_MODEL

logger = logging.getLogger(__name__)

DEFAULT_FORMAT_TEMPLATE = "{original_filename}.upscaled{file_extension}"


@click.argument("image_filepaths", nargs=-1, required=False)
@click.option(
    "--outdir",
    default="./outputs/upscaled",
    show_default=True,
    type=click.Path(),
    help="Where to write results to. Default will be where the directory of the original file.",
)
@click.option("--fix-faces", is_flag=True)
@click.option(
    "--fix-faces-fidelity",
    default=1,
    type=float,
    help="How faithful to the original should face enhancement be. 1 = best fidelity, 0 = best looking face.",
)
@click.option(
    "--upscale-model",
    multiple=True,
    type=str,
    default=[DEFAULT_UPSCALE_MODEL],
    show_default=True,
    help="Specify one or more upscale models to use.",
)
@click.option("--list-models", is_flag=True, help="View available upscale models.")
@click.option(
    "--format",
    "format_template",
    default="{original_filename}.upscaled{file_extension}",
    type=str,
    help="Formats the file name. Default value will save '{original_filename}.upscaled{file_extension}' to the original directory."
    "  {original_filename}: original name without the extension;"
    "{file_sequence_number:pad}: sequence number in directory, can make zero-padded (e.g., 06 for six digits).;"
    " {algorithm}: upscaling algorithm; "
    "{now:%Y-%m-%d:%H-%M-%S}: current date and time, customizable using standard strftime format codes. "
    "Use 'DEV' to config to save in standard imaginAIry format '{file_sequence_number:06}_{algorithm}_{original_filename}.upscaled{file_extension}'. ",
)
@click.command("upscale")
def upscale_cmd(
    image_filepaths,
    outdir,
    fix_faces,
    fix_faces_fidelity,
    upscale_model,
    list_models,
    format_template,
):
    """
    Upscale an image 4x using AI.
    """

    from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
    from imaginairy.enhancers.upscale import upscale_image, upscale_model_lookup
    from imaginairy.schema import LazyLoadingImage
    from imaginairy.utils import glob_expand_paths
    from imaginairy.utils.format_file_name import format_filename, get_url_file_name
    from imaginairy.utils.log_utils import configure_logging

    configure_logging()

    if list_models:
        for model_name in upscale_model_lookup:
            click.echo(f"{model_name}")
        return

    os.makedirs(outdir, exist_ok=True)
    image_filepaths = glob_expand_paths(image_filepaths)

    if not image_filepaths:
        click.echo(
            "Error: No valid image file paths found. Please check the provided file paths."
        )
        return

    if format_template == "DEV":
        format_template = "{file_sequence_number:06}_{algorithm}_{original_filename}.upscaled{file_extension}"
    elif format_template == "DEFAULT":
        format_template = DEFAULT_FORMAT_TEMPLATE

    for n, p in enumerate(image_filepaths):
        if p.startswith("http"):
            img = LazyLoadingImage(url=p)
        else:
            img = LazyLoadingImage(filepath=p)
        orig_height = img.height
        for model in upscale_model:
            logger.info(
                f"Upscaling ({n + 1}/{len(image_filepaths)}) {p} ({img.width}x{img.height})..."
            )

            img = upscale_image(img, model)
            if fix_faces:
                img = enhance_faces(img, fidelity=fix_faces_fidelity)

            if format_template == DEFAULT_FORMAT_TEMPLATE:
                outdir = os.path.dirname(p) + "/"

            file_base_name, extension = os.path.splitext(os.path.basename(p))
            base_count = len(os.listdir(outdir))

            now = datetime.now(timezone.utc)

            if model.startswith(("https://", "http://")):
                model_name = get_url_file_name(model)
            else:
                model_name = model

            new_file_name_data = {
                "original_filename": file_base_name,
                "output_path": outdir,
                "file_sequence_number": base_count,
                "algorithm": model_name,
                "now": now,
                "file_extension": extension,
            }
            new_file_name = format_filename(format_template, new_file_name_data)
            new_file_path = os.path.join(outdir, new_file_name)
            img.save(new_file_path)
            scale = int(img.height / orig_height)
            logger.info(
                f"Upscaled {scale}x to {img.width}x{img.height} and saved to {new_file_path}"
            )

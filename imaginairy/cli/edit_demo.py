import click


@click.command("edit-demo")
@click.argument("image_paths", metavar="PATH|URL", required=True, nargs=-1)
@click.option(
    "--outdir",
    default="./outputs",
    show_default=True,
    type=click.Path(),
    help="Where to write results to.",
)
@click.option(
    "-h",
    "--height",
    default=512,
    show_default=True,
    type=int,
    help="Image height. Should be multiple of 8.",
)
@click.option(
    "-w",
    "--width",
    default=512,
    show_default=True,
    type=int,
    help="Image width. Should be multiple of 8.",
)
def edit_demo_cmd(image_paths, outdir, height, width):
    """Make some fun pre-set edits to input photos."""
    from imaginairy.cli.shared import imaginairy_click_context
    from imaginairy.surprise_me import create_surprise_me_images

    with imaginairy_click_context():
        for image_path in image_paths:
            create_surprise_me_images(
                image_path,
                outdir=outdir,
                make_gif=True,
                width=width,
                height=height,
                seed=1,
            )

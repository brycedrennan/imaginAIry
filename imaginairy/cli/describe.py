import click


@click.argument("image_filepaths", nargs=-1)
@click.command()
def describe_cmd(image_filepaths):
    """Generate text descriptions of images."""
    import os

    from imaginairy import LazyLoadingImage
    from imaginairy.enhancers.describe_image_blip import generate_caption

    imgs = []
    for p in image_filepaths:
        if p.startswith("http"):
            img = LazyLoadingImage(url=p)
        elif os.path.isdir(p):
            print(f"Skipping directory: {p}")
            continue
        else:
            img = LazyLoadingImage(filepath=p)
        imgs.append(img)
    for img in imgs:
        print(generate_caption(img.copy()))

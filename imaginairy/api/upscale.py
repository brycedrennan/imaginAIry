from typing import TYPE_CHECKING, Union

from imaginairy.config import DEFAULT_UPSCALE_MODEL

if TYPE_CHECKING:
    from PIL import Image

    from imaginairy.schema import LazyLoadingImage


def upscale_image(
    img: "Union[LazyLoadingImage, Image.Image, str]",
    upscale_model: str = DEFAULT_UPSCALE_MODEL,
    tile_size: int = 512,
    tile_pad: int = 50,
    repetition: int = 1,
    device=None,
) -> "Image.Image":
    """
    Upscales an image using a specified super-resolution model.

    It accepts an image in various forms: a LazyLoadingImage instance, a PIL Image,
    or a string representing a URL or file path. Supports different upscaling models, customizable tile size, padding,
    and the number of repetitions for upscaling. It can use tiles to manage memory usage on large images and supports multiple passes for upscaling.

    Args:
        img (LazyLoadingImage | Image.Image | str): The input image.
        upscale_model (str, optional): Upscaling model to use. Defaults to realesrgan-x2-plus
        tile_size (int, optional): Size of the tiles used for processing the image. Defaults to 512.
        tile_pad (int, optional): Padding size for each tile. Defaults to 50.
        repetition (int, optional): Number of times the upscaling is repeated. Defaults to 1.
        device: The device (CPU/GPU) to be used for computation. If None, the best available device is used.

    Returns:
        Image.Image: The upscaled image as a PIL Image object.
    """
    from PIL import Image

    from imaginairy.enhancers.upscale import upscale_image
    from imaginairy.schema import LazyLoadingImage

    if isinstance(img, str):
        if img.startswith("https://"):
            img = LazyLoadingImage(url=img)
        else:
            img = LazyLoadingImage(filepath=img)
    elif isinstance(img, Image.Image):
        img = LazyLoadingImage(img=img)

    return upscale_image(
        img,
        upscale_model,
        tile_size=tile_size,
        tile_pad=tile_pad,
        repetition=repetition,
        device=device,
    )

import logging
from typing import TYPE_CHECKING, Union

from imaginairy.config import DEFAULT_UPSCALE_MODEL
from imaginairy.utils import get_device

if TYPE_CHECKING:
    from PIL import Image

    from imaginairy.schema import LazyLoadingImage


upscale_model_lookup = {
    # RealESRGAN
    "ultrasharp": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/1856559b50de25116a7c07261177dd128f1f5664/4x-UltraSharp.pth",
    "realesrgan-x4-plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan-x2-plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    # ESRGAN
    "esrgan-x4": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    # HAT
    "real-hat": "https://huggingface.co/imaginairy/model-weights/resolve/main/weights/super-resolution/hat/Real_HAT_GAN_SRx4.safetensors",
    "real-hat-sharper": "https://huggingface.co/imaginairy/model-weights/resolve/main/weights/super-resolution/hat/Real_HAT_GAN_sharper.safetensors",
    "4xNomos8kHAT-L": "https://huggingface.co/imaginairy/model-weights/resolve/main/weights/super-resolution/hat/4xNomos8kHAT-L_otf.safetensors",
}
logger = logging.getLogger(__name__)


def upscale_image(
    img: "Union[LazyLoadingImage, Image.Image]",
    upscaler_model: str = DEFAULT_UPSCALE_MODEL,
    tile_size: int = 512,
    tile_pad: int = 50,
    repetition: int = 1,
    device=None,
) -> "Image.Image":
    """
    Upscales an image using a specified super-resolution model.

    Supports various upscaling models defined in the `upscale_model_lookup` dictionary, as well as direct URLs to models.
    It can process the image in tiles (to manage memory usage on large images) and supports multiple passes for upscaling.

    Args:
        img (LazyLoadingImage | Image.Image): The input image to be upscaled.
        upscaler_model (str, optional): Key for the upscaling model to use. Defaults to DEFAULT_UPSCALE_MODEL.
        tile_size (int, optional): Size of the tiles used for processing the image. Defaults to 512.
        tile_pad (int, optional): Padding size for each tile. Defaults to 50.
        repetition (int, optional): Number of times the upscaling is repeated. Defaults to 1.
        device: The device (CPU/GPU) to be used for computation. If None, the best available device is used.

    Returns:
        Image.Image: The upscaled image as a PIL Image object.
    """
    import torch
    import torchvision.transforms.functional as F
    from spandrel import ImageModelDescriptor, ModelLoader

    from imaginairy.utils.downloads import get_cached_url_path
    from imaginairy.utils.tile_up import tile_process

    device = device or get_device()

    if upscaler_model in upscale_model_lookup:
        model_url = upscale_model_lookup[upscaler_model]
        model_path = get_cached_url_path(model_url)
    elif upscaler_model.startswith(("https://", "http://")):
        model_url = upscaler_model
        model_path = get_cached_url_path(model_url)
    else:
        model_path = upscaler_model

    model = ModelLoader().load_from_file(model_path)
    logger.debug(f"Upscaling image with model {model.architecture}@{upscaler_model}")

    assert isinstance(model, ImageModelDescriptor)

    model.to(torch.device(device)).eval()

    image_tensor = load_image(img).to(device)
    with torch.no_grad():
        for _ in range(repetition):
            if tile_size > 0:
                image_tensor = tile_process(
                    image_tensor,
                    scale=model.scale,
                    model=model,
                    tile_size=tile_size,
                    tile_pad=tile_pad,
                )
            else:
                image_tensor = model(image_tensor)

    image_tensor = image_tensor.squeeze(0)
    image = F.to_pil_image(image_tensor)
    image = image.resize((img.width * model.scale, img.height * model.scale))

    return image


def load_image(img: "Union[LazyLoadingImage, Image.Image]"):
    """
    Converts a LazyLoadingImage or PIL Image into a PyTorch tensor.
    """
    from torchvision import transforms

    from imaginairy.schema import LazyLoadingImage

    if isinstance(img, LazyLoadingImage):
        img = img.as_pillow()
    transform = transforms.ToTensor()
    image_tensor = transform(img)

    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor.to(get_device())

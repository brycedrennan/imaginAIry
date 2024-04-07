"""
image utils.
"""

import PIL
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.nn import functional as F

from imaginairy.schema import LazyLoadingImage
from imaginairy.utils import img_convert
from imaginairy.utils.mask_helpers import binary_erosion
from imaginairy.utils.mathy import make_odd
from imaginairy.utils.named_resolutions import normalize_image_size
from imaginairy.utils.paths import PKG_ROOT


def pillow_fit_image_within(
    image: PIL.Image.Image | LazyLoadingImage,
    max_height=512,
    max_width=512,
    convert="RGB",
    snap_size=8,
) -> PIL.Image.Image:
    image = image.convert(convert)
    w, h = image.size
    resize_ratio = 1
    if w > max_width or h > max_height:
        resize_ratio = min(max_width / w, max_height / h)
    elif w < max_width and h < max_height:
        # it's smaller than our target image, enlarge
        resize_ratio = max(max_width / w, max_height / h)

    if resize_ratio != 1:
        w, h = int(w * resize_ratio), int(h * resize_ratio)
    # resize to integer multiple of snap_size
    w -= w % snap_size
    h -= h % snap_size

    if (w, h) != image.size:
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    return image


def imgpaths_to_imgs(imgpaths):
    imgs = []
    for imgpath in imgpaths:
        if isinstance(imgpath, str):
            img = LazyLoadingImage(filepath=imgpath)
            imgs.append(img)
        else:
            imgs.append(imgpath)

    return imgs


def add_caption_to_image(
    img: PIL.Image.Image | LazyLoadingImage,
    caption,
    font_size=16,
    font_path=f"{PKG_ROOT}/data/DejaVuSans.ttf",
):
    img_pil = img.as_pillow() if isinstance(img, LazyLoadingImage) else img
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(font_path, font_size)

    x = 15
    y = img_pil.height - 15 - font_size

    draw.text(
        (x, y),
        caption,
        font=font,
        fill=(255, 255, 255),
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )


def create_halo_effect(
    bw_image: PIL.Image.Image, background_color: tuple
) -> PIL.Image.Image:
    from PIL import Image, ImageFilter

    # Step 1: Make white portion of the image transparent
    transparent_image = bw_image.convert("RGBA")
    datas = transparent_image.getdata()
    new_data = []
    for item in datas:
        # Change all white (also shades of whites)
        # to transparent
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    transparent_image.putdata(new_data)  # type: ignore

    # Step 2: Make a copy of the image
    eroded_image = transparent_image.copy()

    # Step 3: Erode and blur the copy
    # eroded_image = ImageOps.invert(eroded_image.convert("L")).convert("1")
    # eroded_image = eroded_image.filter(ImageFilter.MinFilter(3))  # Erode
    eroded_image = eroded_image.filter(ImageFilter.GaussianBlur(radius=25))

    # Step 4: Create new canvas
    new_canvas = Image.new("RGBA", bw_image.size, color=background_color)

    # Step 5: Paste the blurred copy on the new canvas
    new_canvas.paste(eroded_image, (0, 0), eroded_image)

    # Step 6: Paste the original sharp image on the new canvas
    new_canvas.paste(transparent_image, (0, 0), transparent_image)

    return new_canvas


def combine_image(original_img, generated_img, mask_img):
    """Combine the generated image with the original image using the mask image."""
    from PIL import Image

    from imaginairy.utils.log_utils import log_img

    generated_img = generated_img.resize(
        original_img.size,
        resample=Image.Resampling.LANCZOS,
    )

    mask_for_orig_size = mask_img.resize(
        original_img.size,
        resample=Image.Resampling.LANCZOS,
    )
    log_img(mask_for_orig_size, "mask for original image size")

    rebuilt_orig_img = Image.composite(
        original_img,
        generated_img,
        mask_for_orig_size,
    )
    return rebuilt_orig_img


def combine_img_torch(
    target_img: torch.Tensor,
    source_img: torch.Tensor,
    mask_img: torch.Tensor,
) -> torch.Tensor:
    """Combine the source image with the target image using the mask image."""
    img_convert.assert_b1c3hw(target_img)
    img_convert.assert_b1c3hw(source_img)
    img_convert.assert_torch_mask(mask_img)

    # assert mask and img are the same size
    if mask_img.shape[-2:] != source_img.shape[-2:]:
        msg = "Mask and image must have the same height and width."
        raise ValueError(msg)

    # Using the mask, combine the images
    combined_img = target_img * (1 - mask_img) + source_img * mask_img

    img_convert.assert_b1c3hw(combined_img)
    return combined_img


def calc_scale_to_fit_within(height: int, width: int, max_size) -> float:
    max_width, max_height = normalize_image_size(max_size)
    if width <= max_width and height <= max_height:
        return 1

    width_ratio = max_width / width
    height_ratio = max_height / height

    return min(width_ratio, height_ratio)


def aspect_ratio(width, height):
    """
    Calculate the aspect ratio of a given width and height.

    Args:
    width (int): The width dimension.
    height (int): The height dimension.

    Returns:
    str: The aspect ratio in the format 'X:Y'.
    """
    from math import gcd

    # Calculate the greatest common divisor
    divisor = gcd(width, height)

    # Calculate the aspect ratio
    x = width // divisor
    y = height // divisor

    return f"{x}:{y}"


def blur_fill(image: torch.Tensor, mask: torch.Tensor, blur: int, falloff: int):
    blur = make_odd(blur)
    falloff = min(make_odd(falloff), blur - 2)

    original = image.clone()
    alpha = mask.floor()
    if falloff > 0:
        erosion = binary_erosion(alpha, falloff)
        alpha = alpha * gaussian_blur(erosion, falloff)
    alpha = alpha.repeat(1, 3, 1, 1)

    image = gaussian_blur(image, blur)
    image = original + (image - original) * alpha
    return image


def gaussian_blur(image: Tensor, radius: int, sigma: float = 0):
    c = image.shape[-3]
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8

    kernel = _gaussian_kernel(radius, sigma).to(image.device)
    kernel_x = kernel[..., None, :].repeat(c, 1, 1).unsqueeze(1)
    kernel_y = kernel[..., None].repeat(c, 1, 1).unsqueeze(1)

    image = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    image = F.conv2d(image, kernel_x, groups=c)
    image = F.conv2d(image, kernel_y, groups=c)
    return image


def _gaussian_kernel(radius: int, sigma: float):
    x = torch.linspace(-radius, radius, steps=radius * 2 + 1)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()

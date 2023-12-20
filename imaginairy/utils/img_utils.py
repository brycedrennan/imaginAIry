"""
image utils.

Library format cheat sheet:

Library     Dim Order       Channel Order       Value Range     Type
Pillow                      R, G, B, A          0-255           PIL.Image.Image
OpenCV                      B, G, R, A          0-255           np.ndarray
Torch       (B), C, H, W    R, G, B             -1.0-1.0        torch.Tensor

"""
from typing import Sequence

import numpy as np
import PIL
import torch
from einops import rearrange, repeat
from PIL import Image, ImageDraw, ImageFont

from imaginairy.schema import LazyLoadingImage
from imaginairy.utils import get_device
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


def pillow_img_to_torch_image(
    img: PIL.Image.Image | LazyLoadingImage, convert="RGB"
) -> torch.Tensor:
    if convert:
        img = img.convert(convert)
    img_np = np.array(img).astype(np.float32) / 255.0
    # b, h, w, c => b, c, h, w
    img_np = img_np[None].transpose(0, 3, 1, 2)
    img_t = torch.from_numpy(img_np)
    return 2.0 * img_t - 1.0


def pillow_mask_to_latent_mask(
    mask_img: PIL.Image.Image | LazyLoadingImage, downsampling_factor
) -> torch.Tensor:
    mask_img = mask_img.resize(
        (
            mask_img.width // downsampling_factor,
            mask_img.height // downsampling_factor,
        ),
        resample=Image.Resampling.LANCZOS,
    )

    mask = np.array(mask_img).astype(np.float32) / 255.0
    mask = mask[None, None]
    mask_t = torch.from_numpy(mask)
    return mask_t


def pillow_img_to_opencv_img(img: PIL.Image.Image | LazyLoadingImage):
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def torch_image_to_openvcv_img(img: torch.Tensor) -> np.ndarray:
    img = (img + 1) / 2
    img_np = img.detach().cpu().numpy()
    # assert there is only one image
    assert img_np.shape[0] == 1
    img_np = img_np[0]
    img_np = img_np.transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    # RGB to BGR
    img_np = img_np[:, :, ::-1]
    return img_np


def torch_img_to_pillow_img(img_t: torch.Tensor) -> PIL.Image.Image:
    img_t = img_t.to(torch.float32).detach().cpu()
    if len(img_t.shape) == 3:
        img_t = img_t.unsqueeze(0)
    if img_t.shape[0] != 1:
        raise ValueError("Only batch size 1 supported")
    if img_t.shape[1] == 1:
        colorspace = "L"
    elif img_t.shape[1] == 3:
        colorspace = "RGB"
    else:
        msg = (
            f"Unsupported colorspace. {img_t.shape[1]} channels in {img_t.shape} shape"
        )
        raise ValueError(msg)
    img_t = rearrange(img_t, "b c h w -> b h w c")
    img_t = torch.clamp((img_t + 1.0) / 2.0, min=0.0, max=1.0)
    img_np = (255.0 * img_t).cpu().numpy().astype(np.uint8)[0]
    if colorspace == "L":
        img_np = img_np[:, :, 0]
    return Image.fromarray(img_np, colorspace)


def model_latent_to_pillow_img(latent: torch.Tensor) -> PIL.Image.Image:
    from imaginairy.utils.model_manager import get_current_diffusion_model

    if len(latent.shape) == 3:
        latent = latent.unsqueeze(0)
    if latent.shape[0] != 1:
        raise ValueError("Only batch size 1 supported")
    model = get_current_diffusion_model()
    img_t = model.lda.decode(latent)
    return torch_img_to_pillow_img(img_t)


def model_latents_to_pillow_imgs(latents: torch.Tensor) -> Sequence[PIL.Image.Image]:
    return [model_latent_to_pillow_img(latent) for latent in latents]


def pillow_img_to_model_latent(
    model, img: PIL.Image.Image | LazyLoadingImage, batch_size=1, half=True
):
    init_image = pillow_img_to_torch_image(img).to(get_device())
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    if half:
        return model.get_first_stage_encoding(
            model.encode_first_stage(init_image.half())
        )
    return model.get_first_stage_encoding(model.encode_first_stage(init_image))


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


def calc_scale_to_fit_within(height: int, width: int, max_size) -> float:
    max_width, max_height = normalize_image_size(max_size)
    if width <= max_width and height <= max_height:
        return 1

    width_ratio = max_width / width
    height_ratio = max_height / height

    return min(width_ratio, height_ratio)

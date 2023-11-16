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

from imaginairy.paths import PKG_ROOT
from imaginairy.schema import LazyLoadingImage
from imaginairy.utils import get_device


def pillow_fit_image_within(
    image: PIL.Image.Image, max_height=512, max_width=512, convert="RGB", snap_size=8
):
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


def pillow_img_to_torch_image(img: PIL.Image.Image, convert="RGB"):
    if convert:
        img = img.convert(convert)
    img = np.array(img).astype(np.float32) / 255.0
    # b, h, w, c => b, c, h, w
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    return 2.0 * img - 1.0


def pillow_mask_to_latent_mask(mask_img: PIL.Image.Image, downsampling_factor):
    mask_img = mask_img.resize(
        (
            mask_img.width // downsampling_factor,
            mask_img.height // downsampling_factor,
        ),
        resample=Image.Resampling.LANCZOS,
    )

    mask = np.array(mask_img).astype(np.float32) / 255.0
    mask = mask[None, None]
    mask = torch.from_numpy(mask)
    return mask


def pillow_img_to_opencv_img(img: PIL.Image.Image):
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def torch_image_to_openvcv_img(img: torch.Tensor):
    img = (img + 1) / 2
    img = img.detach().cpu().numpy()
    # assert there is only one image
    assert img.shape[0] == 1
    img = img[0]
    img = img.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    # RGB to BGR
    img = img[:, :, ::-1]
    return img


def torch_img_to_pillow_img(img_t: torch.Tensor):
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
    from imaginairy.model_manager import get_current_diffusion_model

    if len(latent.shape) == 3:
        latent = latent.unsqueeze(0)
    if latent.shape[0] != 1:
        raise ValueError("Only batch size 1 supported")
    model = get_current_diffusion_model()
    img_t = model.lda.decode(latent)
    return torch_img_to_pillow_img(img_t)


def model_latents_to_pillow_imgs(latents: torch.Tensor) -> Sequence[PIL.Image.Image]:
    return [model_latent_to_pillow_img(latent) for latent in latents]


def pillow_img_to_model_latent(model, img, batch_size=1, half=True):
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
    img, caption, font_size=16, font_path=f"{PKG_ROOT}/data/DejaVuSans.ttf"
):
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(font_path, font_size)

    x = 15
    y = img.height - 15 - font_size

    draw.text(
        (x, y),
        caption,
        font=font,
        fill=(255, 255, 255),
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )

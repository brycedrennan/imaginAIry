"""
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
from PIL import Image
from torch import Tensor

from imaginairy.schema import LazyLoadingImage
from imaginairy.utils import get_device


def assert_bc3hw(t: Tensor):
    assert isinstance(t, torch.Tensor)
    assert t.ndim == 4
    assert t.shape[1] == 3


def assert_b1c3hw(t: Tensor):
    if not isinstance(t, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    if t.ndim != 4:
        msg = f"Expected 4 dimensions (Batch, Channel, Height, Width), got {t.ndim}"
        raise ValueError(msg)
    if t.shape[1] != 3:
        msg = f"Expected 3 channels, got {t.shape[1]}"
        raise ValueError(msg)


def assert_torch_mask(t: Tensor):
    if not isinstance(t, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    if t.ndim != 4:
        msg = f"Expected 4 dimensions (Batch, Channel, Height, Width), got {t.ndim}"
        raise ValueError(msg)
    if t.shape[1] != 1:
        msg = f"Expected 1 channels, got {t.shape[1]}"
        raise ValueError(msg)


def pillow_img_to_torch_image(
    img: PIL.Image.Image | LazyLoadingImage, convert="RGB"
) -> torch.Tensor:
    if convert:
        img = img.convert(convert)
    img_np = np.array(img).astype(np.float32) / 255.0

    if len(img_np.shape) == 2:
        # add channel at end if missing
        img_np = img_np[:, :, None]
    # b, h, w, c => b, c, h, w
    img_np = img_np[None].transpose(0, 3, 1, 2)
    img_t = torch.from_numpy(img_np)
    return 2.0 * img_t - 1.0


def pillow_mask_255_to_torch_mask(
    mask: PIL.Image.Image | LazyLoadingImage,
) -> torch.Tensor:
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_np = mask_np[None, None]
    mask_t = torch.from_numpy(mask_np)
    return mask_t


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


def assert_ndarray_uint8_255_hwc(img):
    # assert input_image is ndarray with colors 0-255
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.max() <= 255
    assert img.min() >= 0


def assert_tensor_uint8_255_bchw(img):
    # assert input_image is a PyTorch tensor with colors 0-255 and dimensions (C, H, W)
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.uint8
    assert img.ndim == 4
    assert img.shape[1] == 3
    assert img.max() <= 255
    assert img.min() >= 0


def assert_tensor_float_11_bchw(img):
    # assert input_image is a PyTorch tensor with colors -1 to 1 and dimensions (C, H, W)
    if not isinstance(img, torch.Tensor):
        msg = f"Input image must be a PyTorch tensor, but got {type(img)}"
        raise TypeError(msg)

    if img.dtype not in (torch.float32, torch.float64, torch.float16):
        msg = f"Input image must be a float tensor, but got {img.dtype}"
        raise ValueError(msg)

    if img.ndim != 4:
        msg = f"Input image must be 4D (B, C, H, W), but got {img.ndim}D"
        raise ValueError(msg)

    if img.shape[1] != 3:
        msg = f"Input image must have 3 channels, but got {img.shape[1]}"
        raise ValueError(msg)
    if img.max() > 1 or img.min() < -1:
        msg = f"Input image must have values in [-1, 1], but got {img.min()} .. {img.max()}"
        raise ValueError(msg)

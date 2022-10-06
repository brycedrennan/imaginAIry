from typing import Sequence

import numpy as np
import PIL
import torch
from einops import rearrange, repeat
from PIL import Image

from imaginairy.utils import get_device


def pillow_fit_image_within(image: PIL.Image.Image, max_height=512, max_width=512):
    image = image.convert("RGB")
    w, h = image.size
    resize_ratio = 1
    if w > max_width or h > max_height:
        resize_ratio = min(max_width / w, max_height / h)
    elif w < max_width and h < max_height:
        # it's smaller than our target image, enlarge
        resize_ratio = max(max_width / w, max_height / h)

    if resize_ratio != 1:
        w, h = int(w * resize_ratio), int(h * resize_ratio)
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    if (w, h) != image.size:
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    return image


def pillow_img_to_torch_image(img: PIL.Image.Image):
    img = img.convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    return 2.0 * img - 1.0


def pillow_img_to_opencv_img(img: PIL.Image.Image):
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def model_latents_to_pillow_imgs(latents: torch.Tensor) -> Sequence[PIL.Image.Image]:
    from imaginairy.api import load_model  # noqa

    model = load_model()
    latents = model.decode_first_stage(latents)
    latents = torch.clamp((latents + 1.0) / 2.0, min=0.0, max=1.0)
    imgs = []
    for latent in latents:
        latent = 255.0 * rearrange(latent.cpu().numpy(), "c h w -> h w c")
        img = Image.fromarray(latent.astype(np.uint8))
        imgs.append(img)
    return imgs


def pillow_img_to_model_latent(model, img, batch_size=1, half=True):
    # init_image = pil_img_to_torch(img, half=half).to(device)
    init_image = pillow_img_to_torch_image(img).to(get_device())
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    if half:
        return model.get_first_stage_encoding(
            model.encode_first_stage(init_image.half())
        )
    return model.get_first_stage_encoding(model.encode_first_stage(init_image))

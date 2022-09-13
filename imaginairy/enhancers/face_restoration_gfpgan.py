from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from imaginairy.utils import get_cached_url_path, get_device


@lru_cache()
def face_enhance_model(model_type="codeformer"):
    from gfpgan import GFPGANer

    if model_type == "gfpgan":
        arch = "clean"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    elif model_type == "codeformer":
        arch = "CodeFormer"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth"
    else:
        raise ValueError("model_type must be one of gfpgan, codeformer")

    model_path = get_cached_url_path(url)

    if get_device() == "cuda":
        device = "cuda"
    else:
        device = "cpu"

    return GFPGANer(
        model_path=model_path,
        upscale=1,
        arch=arch,
        channel_multiplier=2,
        bg_upsampler=None,
        device=device,
    )


def fix_faces_gfpgan(image, model_type):
    image = image.convert("RGB")
    np_img = np.array(image, dtype=np.uint8)
    cropped_faces, restored_faces, restored_img = face_enhance_model(
        model_type
    ).enhance(
        np_img, has_aligned=False, only_center_face=False, paste_back=True, weight=0
    )
    res = Image.fromarray(restored_img)

    return res

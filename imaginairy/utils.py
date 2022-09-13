import importlib
import logging
import os.path
import platform
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Optional

import numpy as np
import requests
import torch
from PIL import Image
from torch import Tensor
from torch.overrides import handle_torch_function, has_torch_function_variadic
from transformers import cached_path

logger = logging.getLogger(__name__)


@lru_cache()
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@lru_cache()
def get_device_name(device_type):
    if device_type == "cuda":
        return torch.cuda.get_device_name(0)
    return platform.processor()


def log_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    logger.debug(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _fixed_layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Applies Layer Normalization for last certain number of dimensions.
    See :class:`~torch.nn.LayerNorm` for details.
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(
            _fixed_layer_norm,
            (input, weight, bias),
            input,
            normalized_shape,
            weight=weight,
            bias=bias,
            eps=eps,
        )
    return torch.layer_norm(
        input.contiguous(),
        normalized_shape,
        weight,
        bias,
        eps,
        torch.backends.cudnn.enabled,
    )


@contextmanager
def fix_torch_nn_layer_norm():
    """https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1221416526"""
    from torch.nn import functional

    orig_function = functional.layer_norm
    functional.layer_norm = _fixed_layer_norm
    try:
        yield
    finally:
        functional.layer_norm = orig_function


def img_path_to_torch_image(path, max_height=512, max_width=512):
    image = Image.open(path).convert("RGB")
    logger.info(f"Loaded input 🖼 of size {image.size} from {path}")
    return pillow_img_to_torch_image(image, max_height=max_height, max_width=max_width)


def pillow_img_to_torch_image(image, max_height=512, max_width=512):
    w, h = image.size
    resize_ratio = min(max_width / w, max_height / h)
    w, h = int(w * resize_ratio), int(h * resize_ratio)
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0, w, h


def get_cache_dir():
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", None)
    if xdg_cache_home is None:
        user_home = os.getenv("HOME", None)
        if user_home:
            xdg_cache_home = os.path.join(user_home, ".cache")

    if xdg_cache_home is not None:
        return os.path.join(xdg_cache_home, "imaginairy", "weights")

    return os.path.join(os.path.dirname(__file__), ".cached-downloads")


def get_cached_url_path(url):
    try:
        return cached_path(url)
    except OSError:
        pass
    filename = url.split("/")[-1]
    dest = get_cache_dir()
    os.makedirs(dest, exist_ok=True)
    dest_path = os.path.join(dest, filename)
    if os.path.exists(dest_path):
        return dest_path
    r = requests.get(url)

    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path

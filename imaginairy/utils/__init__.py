import importlib
import logging
import platform
import time
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any, List, Optional, Union

import torch
from torch import Tensor, autocast
from torch.nn import functional
from torch.overrides import handle_torch_function, has_torch_function_variadic

logger = logging.getLogger(__name__)


@lru_cache()
def get_device() -> str:
    """Return the best torch backend available."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps:0"

    return "cpu"


@lru_cache()
def get_hardware_description(device_type: str) -> str:
    """Description of the hardware being used."""
    desc = platform.platform()
    if device_type == "cuda":
        desc += "-" + torch.cuda.get_device_name(0)

    return desc


def get_obj_from_str(import_path: str, reload=False) -> Any:
    """
    Gets a python object from a string reference if it's location.

    Example: "functools.lru_cache"
    """
    module_path, obj_name = import_path.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module_path)
        importlib.reload(module_imp)
    module = importlib.import_module(module_path, package=None)
    return getattr(module, obj_name)


def instantiate_from_config(config: Union[dict, str]) -> Any:
    """Instantiate an object from a config dict."""
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        if config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    params = config.get("params", {})
    _cls = get_obj_from_str(config["target"])
    start = time.perf_counter()
    c = _cls(**params)
    end = time.perf_counter()
    logger.debug(f"Instantiation of {_cls} took {end-start} seconds")
    return c


@contextmanager
def platform_appropriate_autocast(precision="autocast"):
    """
    Allow calculations to run in mixed precision, which can be faster.
    """
    precision_scope = nullcontext
    # autocast not supported on CPU
    # https://github.com/pytorch/pytorch/issues/55374
    # https://github.com/invoke-ai/InvokeAI/pull/518
    if precision == "autocast" and get_device() in ("cuda",):
        precision_scope = autocast
    with precision_scope(get_device()):
        yield


def _fixed_layer_norm(
    input: Tensor,  # noqa
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Applies Layer Normalization for last certain number of dimensions.

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
    """https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1221416526."""
    orig_function = functional.layer_norm
    functional.layer_norm = _fixed_layer_norm
    try:
        yield
    finally:
        functional.layer_norm = orig_function


@contextmanager
def fix_torch_group_norm():
    """
    Patch group_norm to cast the weights to the same type as the inputs.

    From what I can understand all the other repos just switch to full precision instead
    of addressing this.  I think this would make things slower but I'm not sure.

    https://github.com/pytorch/pytorch/pull/81852

    """

    orig_group_norm = functional.group_norm

    def _group_norm_wrapper(
        input: Tensor,  # noqa
        num_groups: int,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5,
    ) -> Tensor:
        if weight is not None and weight.dtype != input.dtype:
            weight = weight.to(input.dtype)
        if bias is not None and bias.dtype != input.dtype:
            bias = bias.to(input.dtype)

        return orig_group_norm(
            input=input, num_groups=num_groups, weight=weight, bias=bias, eps=eps
        )

    functional.group_norm = _group_norm_wrapper
    try:
        yield
    finally:
        functional.group_norm = orig_group_norm


def randn_seeded(seed: int, size: List[int]) -> Tensor:
    """Generate a random tensor with a given seed."""
    from hashlib import md5

    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    noise = torch.randn(
        size,
        device="cpu",
        generator=g_cpu,
    )
    # md5 of the torch tensor `noise`
    torch_md5 = md5(noise.numpy().tobytes()).hexdigest()
    logger.debug(f"Made noise of size {size} from seed {seed}. md5:{torch_md5}")
    return noise


def check_torch_working():
    """Check that torch is working."""
    try:
        torch.randn(1, device=get_device())
    except RuntimeError as e:
        if "CUDA" in str(e):
            raise RuntimeError(
                "CUDA is not working.  Make sure you have a GPU and CUDA installed."
            ) from e
        raise e


def frange(start, stop, step):
    """Range but handles floats."""
    x = start
    while True:
        if x >= stop:
            return
        yield x
        x += step


def shrink_list(items, max_size):
    if len(items) <= max_size:
        return items

    removal_ratio = len(items) / (max_size - 1)
    new_items = {}
    for i, item in enumerate(items):
        new_items[int(i / removal_ratio)] = item
    return [items[0]] + list(new_items.values())


def glob_expand_paths(paths):
    import glob
    import os.path

    expanded_paths = []
    for p in paths:
        if p.startswith("http"):
            expanded_paths.append(p)
        else:
            expanded_paths.extend(glob.glob(os.path.expanduser(p)))
    return expanded_paths


def get_next_filenumber(path):
    """Get the next file number in a directory."""
    import os

    filenames = os.listdir(path)
    if not filenames:
        return 0
    file_count = len(filenames)
    filenames.sort()
    try:
        last_file_name = filenames[-1]
        last_file_num = int(last_file_name.split("_")[0])
    except (ValueError, IndexError):
        last_file_num = 0

    return max(file_count, last_file_num + 1)


def check_torch_version():
    """
    Check that the torch version is compatible with ImaginAIry.

    https://github.com/brycedrennan/imaginAIry/issues/329
    """
    from packaging import version

    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        raise RuntimeError("ImaginAIry is not compatible with torch>=2.0.0")

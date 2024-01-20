from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar

import torch
from jaxtyping import Float
from numpy import array, float32
from PIL import Image
from safetensors import safe_open as _safe_open  # type: ignore
from safetensors.torch import save_file as _save_file  # type: ignore
from torch import (
    Tensor,
    device as Device,
    dtype as DType,
    manual_seed as _manual_seed,  # type: ignore
    no_grad as _no_grad,  # type: ignore
    norm as _norm,  # type: ignore
)
from torch.nn.functional import conv2d, interpolate as _interpolate, pad as _pad  # type: ignore

T = TypeVar("T")
E = TypeVar("E")


def norm(x: Tensor) -> Tensor:
    return _norm(x)  # type: ignore


def manual_seed(seed: int) -> None:
    _manual_seed(seed)


class no_grad(_no_grad):
    def __new__(cls, orig_func: Any | None = None) -> "no_grad":  # type: ignore
        return object.__new__(cls)


def pad(x: Tensor, pad: Iterable[int], value: float = 0.0, mode: str = "constant") -> Tensor:
    return _pad(input=x, pad=pad, value=value, mode=mode)  # type: ignore


def interpolate(x: Tensor, factor: float | torch.Size, mode: str = "nearest") -> Tensor:
    return (
        _interpolate(x, scale_factor=factor, mode=mode)
        if isinstance(factor, float | int)
        else _interpolate(x, size=factor, mode=mode)
    )  # type: ignore


# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
def normalize(
    tensor: Float[Tensor, "*batch channels height width"], mean: list[float], std: list[float]
) -> Float[Tensor, "*batch channels height width"]:
    assert tensor.is_floating_point()
    assert tensor.ndim >= 3

    dtype = tensor.dtype
    pixel_mean = torch.tensor(mean, dtype=dtype, device=tensor.device).view(-1, 1, 1)
    pixel_std = torch.tensor(std, dtype=dtype, device=tensor.device).view(-1, 1, 1)
    if (pixel_std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

    return (tensor - pixel_mean) / pixel_std


# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
def gaussian_blur(
    tensor: Float[Tensor, "*batch channels height width"],
    kernel_size: int | tuple[int, int],
    sigma: float | tuple[float, float] | None = None,
) -> Float[Tensor, "*batch channels height width"]:
    assert torch.is_floating_point(tensor)

    def get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Float[Tensor, "kernel_size"]:
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        return kernel1d

    def get_gaussian_kernel2d(
        kernel_size_x: int, kernel_size_y: int, sigma_x: float, sigma_y: float, dtype: DType, device: Device
    ) -> Float[Tensor, "kernel_size_y kernel_size_x"]:
        kernel1d_x = get_gaussian_kernel1d(kernel_size_x, sigma_x).to(device, dtype=dtype)
        kernel1d_y = get_gaussian_kernel1d(kernel_size_y, sigma_y).to(device, dtype=dtype)
        kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
        return kernel2d

    def default_sigma(kernel_size: int) -> float:
        return kernel_size * 0.15 + 0.35

    if isinstance(kernel_size, int):
        kx, ky = kernel_size, kernel_size
    else:
        kx, ky = kernel_size

    if sigma is None:
        sx, sy = default_sigma(kx), default_sigma(ky)
    elif isinstance(sigma, float):
        sx, sy = sigma, sigma
    else:
        assert isinstance(sigma, tuple)
        sx, sy = sigma

    channels = tensor.shape[-3]
    kernel = get_gaussian_kernel2d(kx, ky, sx, sy, dtype=tensor.dtype, device=tensor.device)
    kernel = kernel.expand(channels, 1, kernel.shape[0], kernel.shape[1])

    # pad = (left, right, top, bottom)
    tensor = pad(tensor, pad=(kx // 2, kx // 2, ky // 2, ky // 2), mode="reflect")
    tensor = conv2d(tensor, weight=kernel, groups=channels)

    return tensor


def image_to_tensor(image: Image.Image, device: Device | str | None = None, dtype: DType | None = None) -> Tensor:
    """
    Convert a PIL Image to a Tensor.

    If the image is in mode `RGB` the tensor will have shape `[3, H, W]`, otherwise
    `[1, H, W]` for mode `L` (grayscale) or `[4, H, W]` for mode `RGBA`.

    Values are clamped to the range `[0, 1]`.
    """
    image_tensor = torch.tensor(array(image).astype(float32) / 255.0, device=device, dtype=dtype)

    match image.mode:
        case "L":
            image_tensor = image_tensor.unsqueeze(0)
        case "RGBA" | "RGB":
            image_tensor = image_tensor.permute(2, 0, 1)
        case _:
            raise ValueError(f"Unsupported image mode: {image.mode}")

    return image_tensor.unsqueeze(0)


def tensor_to_image(tensor: Tensor) -> Image.Image:
    """
    Convert a Tensor to a PIL Image.

    The tensor must have shape `[1, channels, height, width]` where the number of
    channels is either 1 (grayscale) or 3 (RGB) or 4 (RGBA).

    Expected values are in the range `[0, 1]` and are clamped to this range.
    """
    assert tensor.ndim == 4 and tensor.shape[0] == 1, f"Unsupported tensor shape: {tensor.shape}"
    num_channels = tensor.shape[1]
    tensor = tensor.clamp(0, 1).squeeze(0)
    tensor = tensor.to(torch.float32)  # to avoid numpy error with bfloat16

    match num_channels:
        case 1:
            tensor = tensor.squeeze(0)
        case 3 | 4:
            tensor = tensor.permute(1, 2, 0)
        case _:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

    return Image.fromarray((tensor.cpu().numpy() * 255).astype("uint8"))  # type: ignore[reportUnknownType]


def safe_open(
    path: Path | str,
    framework: Literal["pytorch", "tensorflow", "flax", "numpy"],
    device: Device | str = "cpu",
) -> dict[str, Tensor]:
    framework_mapping = {
        "pytorch": "pt",
        "tensorflow": "tf",
        "flax": "flax",
        "numpy": "numpy",
    }
    return _safe_open(str(path), framework=framework_mapping[framework], device=str(device))  # type: ignore


def load_from_safetensors(path: Path | str, device: Device | str = "cpu") -> dict[str, Tensor]:
    with safe_open(path=path, framework="pytorch", device=device) as tensors:  # type: ignore
        return {key: tensors.get_tensor(key) for key in tensors.keys()}  # type: ignore


def load_metadata_from_safetensors(path: Path | str) -> dict[str, str] | None:
    with safe_open(path=path, framework="pytorch") as tensors:  # type: ignore
        return tensors.metadata()  # type: ignore


def save_to_safetensors(path: Path | str, tensors: dict[str, Tensor], metadata: dict[str, str] | None = None) -> None:
    _save_file(tensors, path, metadata)  # type: ignore


def summarize_tensor(tensor: torch.Tensor, /) -> str:
    info_list = [
        f"shape=({', '.join(map(str, tensor.shape))})",
        f"dtype={str(object=tensor.dtype).removeprefix('torch.')}",
        f"device={tensor.device}",
    ]
    if tensor.is_complex():
        tensor_f = tensor.real.float()
    else:
        if tensor.numel() > 0:
            info_list.extend(
                [
                    f"min={tensor.min():.2f}",  # type: ignore
                    f"max={tensor.max():.2f}",  # type: ignore
                ]
            )
        tensor_f = tensor.float()

    info_list.extend(
        [
            f"mean={tensor_f.mean():.2f}",
            f"std={tensor_f.std():.2f}",
            f"norm={norm(x=tensor_f):.2f}",
            f"grad={tensor.requires_grad}",
        ]
    )

    return "Tensor(" + ", ".join(info_list) + ")"

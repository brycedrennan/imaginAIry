from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from imaginairy.utils import img_convert
from imaginairy.utils.img_convert import assert_bc3hw
from imaginairy.utils.mathy import make_odd


def binary_erosion(mask: Tensor, radius: int):
    kernel = torch.ones(1, 1, radius * 2 + 1, radius * 2 + 1, device=mask.device)
    mask = F.pad(mask, (radius, radius, radius, radius), mode="constant", value=1)
    mask = F.conv2d(mask, kernel, groups=1)
    mask = (mask == kernel.numel()).to(mask.dtype)
    return mask


def highlight_masked_area(
    img: Tensor,
    mask: Tensor,
    color: Union[tuple[int, int, int], None] = None,
    highlight_strength: float = 0.5,
) -> Tensor:
    """
    Highlights the masked area of an image tensor with a specified color.
    """
    from imaginairy.utils.img_utils import combine_img_torch

    img_convert.assert_b1c3hw(img)
    img_convert.assert_torch_mask(mask)

    # Ensure mask is in the same device as image_tensor
    mask = mask.to(img.device)
    if color is None:
        color = tuple(np.random.randint(0, 256, 3))
    else:
        if any(c > 255 or c < 0 for c in color):
            raise ValueError("Color values must be in the range [0, 255].")
    # Convert color to a tensor and normalize to [0, 1]
    color_tensor = torch.tensor(color, device=img.device, dtype=img.dtype) / 255.0
    solid_color = torch.ones_like(img)
    for channel in range(3):
        solid_color[:, channel, :, :] *= color_tensor[channel]

    highlighted_image = combine_img_torch(img, solid_color, mask * highlight_strength)

    return highlighted_image


def fill_neutral(image: Tensor, mask: Tensor, falloff: int = 1) -> Tensor:
    img_convert.assert_bc3hw(image)
    img_convert.assert_torch_mask(mask)

    mask = mask_falloff(mask, falloff)
    filled_img = image.detach().clone()
    m = (1.0 - mask).squeeze(0).squeeze(0)
    for i in range(3):
        filled_img[:, i, :, :] -= 0.5
        filled_img[:, i, :, :] *= m
        filled_img[:, i, :, :] += 0.5
    img_convert.assert_bc3hw(filled_img)
    return filled_img


def fill_noise(image: Tensor, mask: Tensor, falloff: int = 1, seed=1) -> Tensor:
    """
    Fills a masked area in an image with random noise.
    """
    img_convert.assert_bc3hw(image)
    img_convert.assert_torch_mask(mask)

    mask = mask_falloff(mask, falloff)
    filled_img = image.detach().clone()
    noise = torch.rand_like(filled_img) * 2 - 1
    filled_img = filled_img * (1 - mask) + noise * mask
    img_convert.assert_bc3hw(filled_img)
    return filled_img


# def expand_mask(mask, expand, tapered_corners):
#     c = 0 if tapered_corners else 1
#     kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
#     mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
#     out = []
#     for m in mask:
#         output = m.numpy()
#         for _ in range(abs(expand)):
#             if expand < 0:
#                 output = scipy.ndimage.grey_erosion(output, footprint=kernel)
#             else:
#                 output = scipy.ndimage.grey_dilation(output, footprint=kernel)
#         output = torch.from_numpy(output)
#         out.append(output)
#     return (torch.stack(out, dim=0),)


def mask_falloff(mask: Tensor, falloff: int) -> Tensor:
    """
    Applies a falloff effect to a binary mask tensor to create smooth transitions at its edges.
    """
    from imaginairy.utils.img_utils import gaussian_blur

    alpha = mask.expand(1, *mask.shape[-3:]).floor()
    if falloff > 0:
        falloff = make_odd(falloff)
        erosion = binary_erosion(alpha, falloff)
        alpha = alpha * gaussian_blur(erosion, falloff)
    return alpha


def fill_navier_stokes(image: Tensor, mask: Tensor, falloff: int = 1) -> Tensor:
    """
    Fills a masked area in an image using Navier-Stokes inpainting.

    https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html
    """
    import cv2

    assert_bc3hw(image)
    alpha = mask_falloff(mask, falloff)
    filled_img = image.detach().clone()

    alpha_np = alpha.squeeze(0).squeeze(0).cpu().numpy()
    alpha_bc = alpha_np.reshape(*alpha_np.shape)
    filled_img = filled_img.squeeze(0)
    for channel_slice in filled_img:
        image_np = channel_slice.cpu().numpy()
        filled_np = cv2.inpaint(
            (255.0 * (image_np + 1) / 2).astype(np.uint8),
            (255.0 * alpha_np).astype(np.uint8),
            3,
            cv2.INPAINT_NS,
        )
        filled_np = (filled_np.astype(np.float32) / 255.0) * 2 - 1
        filled_np = image_np * (1.0 - alpha_bc) + filled_np * alpha_bc
        channel_slice.copy_(torch.from_numpy(filled_np))

    filled_img = filled_img.unsqueeze(0)
    assert_bc3hw(filled_img)
    return filled_img

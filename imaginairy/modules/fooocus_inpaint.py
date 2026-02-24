"""Fooocus inpaint patch for SDXL.

Uses a tiny InpaintHead conv + LoRA patches to enable high-quality inpainting
with standard SDXL weights (no dedicated inpainting model needed).

References:
- https://github.com/Acly/comfyui-inpaint-nodes
- https://huggingface.co/lllyasviel/fooocus_inpaint
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import Tensor, nn

from imaginairy.utils.downloads import get_cached_url_path

logger = logging.getLogger(__name__)

INPAINT_HEAD_URL = "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"
INPAINT_PATCH_V26_URL = "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch"


class InpaintHead(nn.Module):
    """Tiny 3x3 conv: [latent_mask(1ch), latent_pixels(4ch)] -> 320ch features."""

    def __init__(self):
        super().__init__()
        self.head = nn.Parameter(torch.empty(320, 5, 3, 3))

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (1, 1, 1, 1), "replicate")
        return F.conv2d(x, weight=self.head)


@lru_cache(maxsize=1)
def get_inpaint_head(device: str, dtype: torch.dtype) -> InpaintHead:
    head = InpaintHead()
    path = get_cached_url_path(INPAINT_HEAD_URL)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    head.load_state_dict(sd)
    return head.to(device=device, dtype=dtype).eval()


@lru_cache(maxsize=1)
def _load_fooocus_patch_raw():
    path = get_cached_url_path(INPAINT_PATCH_V26_URL, category="weights")
    return torch.load(path, map_location="cpu", weights_only=False)


def _decode_fooocus_weight(w) -> Tensor:
    """Decode quantized weight: (uint8_tensor, min, max) -> float tensor."""
    if isinstance(w, tuple) and len(w) == 3:
        t, w_min, w_max = w
        return (t.float() / 255.0) * (w_max - w_min) + w_min
    if isinstance(w, Tensor):
        return w.float()
    msg = f"Unexpected fooocus weight format: {type(w)}"
    raise ValueError(msg)


def _build_compvis_to_refiners_map() -> dict[str, str]:
    """Build a CompVis UNet key -> refiners parameter name map for SDXL.

    CompVis keys: model.diffusion_model.xxx.weight
    Diffusers keys: time_embedding.linear_1.weight (from CompVis map)
    Refiners map keys: time_embedding.linear_1 (no suffix)
    Refiners param names: TimestepEncoder.Sum.Chain.RangeEncoder.Linear_1.weight
    """
    from imaginairy.weight_management.translators import (
        diffusers_unet_sdxl_to_refiners_translator,
        load_weight_map,
    )

    compvis_to_diffusers = load_weight_map("Compvis-UNet-SDXL-to-Diffusers")
    diffusers_to_refiners = diffusers_unet_sdxl_to_refiners_translator()

    result = {}
    for compvis_key, diffusers_key_full in compvis_to_diffusers.name_map.items():
        # diffusers_key_full has .weight/.bias suffix, strip it for lookup
        for suffix in (".weight", ".bias"):
            if diffusers_key_full.endswith(suffix):
                diffusers_key_base = diffusers_key_full[: -len(suffix)]
                refiners_base = diffusers_to_refiners.name_map.get(diffusers_key_base)
                if refiners_base:
                    # Reconstruct full refiners param name: base + suffix
                    result[compvis_key] = f"{refiners_base}{suffix}"
                break
    return result


def _build_patch_key_map(patch_keys: list[str]) -> dict[str, str]:
    """Map fooocus patch keys to refiners UNet parameter names.

    Patch keys are in CompVis format without 'model.' prefix, e.g.:
      diffusion_model.input_blocks.0.0.weight
    The CompVis weight map expects 'model.' prefix, so we prepend it.
    """
    compvis_to_refiners = _build_compvis_to_refiners_map()

    key_map = {}
    for patch_key in patch_keys:
        # Patch keys are like "diffusion_model.xxx.weight"
        # CompVis map expects "model.diffusion_model.xxx.weight"
        compvis_key = f"model.{patch_key}"
        if compvis_key in compvis_to_refiners:
            key_map[patch_key] = compvis_to_refiners[compvis_key]

    return key_map


def apply_fooocus_patch_to_unet(unet, alpha: float = 1.0):
    """Apply fooocus v26 patch weights to the SDXL UNet.

    Loads the patch file (960 quantized weight deltas in CompVis key format),
    translates keys to refiners format, decodes, and applies as deltas.
    Parameters are cloned before modification to preserve the cached model.
    """
    patch = _load_fooocus_patch_raw()
    key_map = _build_patch_key_map(list(patch.keys()))
    logger.info(f"Fooocus patch: mapped {len(key_map)}/{len(patch)} keys")

    unet_params = dict(unet.named_parameters())

    applied = 0
    skipped = 0
    for patch_key, weight_data in patch.items():
        refiners_key = key_map.get(patch_key)
        if refiners_key is None:
            skipped += 1
            continue

        if refiners_key not in unet_params:
            skipped += 1
            continue

        delta = _decode_fooocus_weight(weight_data)
        param = unet_params[refiners_key]
        if delta.shape != param.shape:
            logger.debug(
                f"Shape mismatch for {patch_key}: {delta.shape} vs {param.shape}"
            )
            skipped += 1
            continue

        _apply_delta(param, delta, alpha)
        applied += 1

    logger.info(f"Applied {applied} fooocus patches, skipped {skipped}")
    return applied


def _apply_delta(param: nn.Parameter, delta: Tensor, alpha: float):
    """Apply a weight delta to a parameter.

    Saves the original data on first call so repeated applications (e.g. across
    --repeats) always start from the true original, avoiding accumulation.
    structural_copy() shares leaf modules with the LRU-cached model, so
    setting param.data affects the cache — _save_original / _restore_original
    handle cleanup.
    """
    if not hasattr(param, "_fooocus_original"):
        param._fooocus_original = param.data.clone()
    param.data = param._fooocus_original.clone()
    param.data += alpha * delta.to(device=param.device, dtype=param.dtype)


def unapply_fooocus_patches(unet):
    """Restore original weights on shared parameters after generation.

    Must be called after each generation to keep the LRU-cached model clean.
    """
    restored = 0
    for param in unet.parameters():
        if hasattr(param, "_fooocus_original"):
            param.data = param._fooocus_original
            del param._fooocus_original
            restored += 1
    if restored:
        logger.debug(f"Restored {restored} fooocus-patched parameters")


def fooocus_fill(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Fill masked region with progressively blurred surroundings.

    Matches Fooocus's exact kernel schedule: cascading box blurs with specific
    repeat counts. After each blur, unmasked (keep) pixels are restored from
    the original, so only the masked region accumulates the blur fill.
    """
    img_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))

    current = img_np.copy()
    # Indices of KEEP pixels (mask < 127 = not inpainted)
    keep_idx = np.where(mask_np < 127)
    keep_vals = img_np[keep_idx]

    for k, repeats in [
        (512, 2),
        (256, 2),
        (128, 4),
        (64, 4),
        (33, 8),
        (15, 8),
        (5, 16),
        (3, 16),
    ]:
        for _ in range(repeats):
            current = np.array(Image.fromarray(current).filter(ImageFilter.BoxBlur(k)))
            current[keep_idx] = keep_vals

    return Image.fromarray(current)


def morphological_open_mask(mask: Image.Image) -> Image.Image:
    """Soften mask edges matching Fooocus's morphological_open.

    Binarizes to 0/256 int16, then 32 iterations of (cv2.dilate - 8).
    Each iteration dilates by 1px but subtracts 8 from the edge,
    creating a gradual falloff from 256→0 over ~32px at boundaries.
    """
    import cv2

    m = np.array(mask.convert("L"))
    x = np.zeros_like(m, dtype=np.int16)
    x[m > 127] = 256

    kernel = np.ones((3, 3), dtype=np.int16)
    for _ in range(32):
        maxed = cv2.dilate(x, kernel) - 8
        x = np.maximum(maxed, x)

    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8), mode="L")


@dataclass
class InpaintCrop:
    """Context for Fooocus-style interested area crop + resize."""

    crop_box: tuple[int, int, int, int]  # (y1, y2, x1, x2) in original coords
    original_image: np.ndarray  # full-size original (H, W, 3)
    working_size: tuple[int, int]  # (W, H) the generation runs at


def _shape_ceil(h: int, w: int) -> float:
    """Fooocus's get_shape_ceil: sqrt(H*W) rounded to nearest 64."""
    import math

    return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0


def _resize_to_ceil(img_np: np.ndarray, target_ceil: float) -> np.ndarray:
    """Resize so sqrt(H*W) ≈ target_ceil, snapping dims to 64px grid."""
    h, w = img_np.shape[:2]
    for _ in range(256):
        cur = _shape_ceil(h, w)
        if abs(cur - target_ceil) < 0.1:
            break
        k = target_ceil / cur
        h = int(round(h * k / 64.0) * 64)
        w = int(round(w * k / 64.0) * 64)
    if h == img_np.shape[0] and w == img_np.shape[1]:
        return img_np
    pil = Image.fromarray(img_np).resize((w, h), resample=Image.Resampling.LANCZOS)
    return np.array(pil)


def prepare_inpaint_crop(
    init_image: Image.Image,
    mask_image: Image.Image,
    k: float = 0.618,
    target_ceil: float = 1024.0,
) -> tuple[Image.Image, Image.Image, InpaintCrop]:
    """Crop to mask's interested area and resize to ~target_ceil resolution.

    Matches Fooocus InpaintWorker: find mask bbox, expand 1.15x square-ish,
    then expand until covering k fraction of each dimension. Resize so
    sqrt(H*W) ≈ target_ceil. Always returns a crop — never skips.
    """
    img_np = np.array(init_image.convert("RGB"))
    mask_np = np.array(mask_image.convert("L"))
    h_full, w_full = img_np.shape[:2]

    # Find mask bbox and compute initial interested area (1.15x expand, square-ish)
    rows, cols = np.where(mask_np > 0)
    if len(rows) == 0:
        # No mask — just use whole image
        a, b, c, d = 0, h_full, 0, w_full
    else:
        a, b = int(rows.min()), int(rows.max())
        c, d = int(cols.min()), int(cols.max())
        # Expand 1.15x centered
        mid_y, mid_x = (a + b) // 2, (c + d) // 2
        half = int(max(b - a, d - c) * 1.15 / 2)
        a, b = mid_y - half, mid_y + half + 1
        c, d = mid_x - half, mid_x + half + 1
        a = max(0, min(a, h_full))
        b = max(0, min(b, h_full))
        c = max(0, min(c, w_full))
        d = max(0, min(d, w_full))

    # Expand until covering k fraction of each dimension
    while True:
        if b - a >= h_full * k and d - c >= w_full * k:
            break
        grow_h = (b - a) < (d - c)
        grow_w = not grow_h
        if b - a >= h_full:
            grow_w = True
        if d - c >= w_full:
            grow_h = True
        if grow_h:
            a = max(0, a - 1)
            b = min(h_full, b + 1)
        if grow_w:
            c = max(0, c - 1)
            d = min(w_full, d + 1)

    # Crop
    cropped_img = img_np[a:b, c:d]
    cropped_mask = mask_np[a:b, c:d]

    # Resize to ~target_ceil
    cropped_img = _resize_to_ceil(cropped_img, target_ceil)
    out_h, out_w = cropped_img.shape[:2]
    # Resize mask to match (nearest for binary mask)
    cropped_mask = np.array(
        Image.fromarray(cropped_mask).resize(
            (out_w, out_h), resample=Image.Resampling.NEAREST
        )
    )
    # Re-binarize mask after resize
    cropped_mask[cropped_mask > 127] = 255
    cropped_mask[cropped_mask <= 127] = 0

    crop = InpaintCrop(
        crop_box=(a, b, c, d),
        original_image=img_np,
        working_size=(out_w, out_h),
    )
    return (
        Image.fromarray(cropped_img),
        Image.fromarray(cropped_mask, mode="L"),
        crop,
    )


def post_process_inpaint(
    crop: InpaintCrop,
    generated: Image.Image,
    morph_mask: Image.Image,
) -> Image.Image:
    """Resize generated back to crop region, paste, blend with morphological mask.

    Matches Fooocus post_process + color_correction.
    """
    a, b, c, d = crop.crop_box
    crop_h, crop_w = b - a, d - c

    # Resize generated to original crop dimensions
    gen_np = np.array(
        generated.resize((crop_w, crop_h), resample=Image.Resampling.LANCZOS)
    )

    # Paste into original
    result = crop.original_image.copy()
    result[a:b, c:d] = gen_np

    # Blend with morphological mask (caller provides full-size morph mask)
    morph_np = np.array(morph_mask)
    if morph_np.shape[:2] != result.shape[:2]:
        morph_np = np.array(
            Image.fromarray(morph_np).resize(
                (result.shape[1], result.shape[0]), resample=Image.Resampling.LANCZOS
            )
        )
    w = morph_np[:, :, None].astype(np.float32) / 255.0
    fg = result.astype(np.float32)
    bg = crop.original_image.astype(np.float32)
    blended = (fg * w + bg * (1.0 - w)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def compute_inpaint_conditioning(
    sd,
    init_image: Image.Image,
    mask_image: Image.Image,
) -> Tensor:
    """Compute InpaintHead features for the fooocus patch method.

    Args:
        sd: The StableDiffusion_XL model
        init_image: Original image (PIL RGB)
        mask_image: Mask image (PIL L) - white = area to inpaint

    Returns:
        Feature tensor (1, 320, H/8, W/8) to inject into UNet first block
    """
    device = sd.unet.device
    dtype = sd.unet.dtype

    mask = mask_image.convert("L")
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_binary = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
    mask_binary = (mask_binary > 0.5).float()

    # Fill masked region: fooocus_fill for color-blended background,
    # then composite mid-gray (128) over masked pixels to match Fooocus's
    # encode_vae_inpaint which fills with 0.5.
    filled = fooocus_fill(init_image.convert("RGB"), mask)
    gray = Image.new("RGB", init_image.size, (128, 128, 128))
    masked_pil = Image.composite(gray, filled, mask)

    latent_pixels = sd.lda.encode_image(masked_pil).to(device=device, dtype=dtype)

    # Downsample mask to latent space
    latent_mask = F.max_pool2d(
        mask_binary.to(device=device, dtype=dtype), kernel_size=8, stride=8
    ).round()

    # Run through InpaintHead
    head = get_inpaint_head(str(device), dtype)
    head_input = torch.cat([latent_mask, latent_pixels], dim=1)
    with torch.no_grad():
        features = head(head_input)

    return features


class AddInpaintFeature(nn.Module):
    """Adds pre-computed InpaintHead features to the UNet's first conv output.

    Supports MultiDiffusion tiling via set_tile/clear_tile.
    """

    def __init__(self, feature: Tensor):
        super().__init__()
        self.register_buffer("feature", feature)
        self._ty = 0
        self._tx = 0
        self._th = feature.shape[2]
        self._tw = feature.shape[3]

    def set_tile(self, ty: int, tx: int, th: int, tw: int):
        self._ty, self._tx, self._th, self._tw = ty, tx, th, tw

    def clear_tile(self):
        self._ty, self._tx = 0, 0
        self._th, self._tw = self.feature.shape[2], self.feature.shape[3]

    def forward(self, x: Tensor) -> Tensor:
        feat = self.feature[
            :, :, self._ty : self._ty + self._th, self._tx : self._tx + self._tw
        ]
        # Handle CFG batch doubling (UNet gets 2x batch for guidance)
        repeat = x.shape[0] // feat.shape[0]
        feat = feat.repeat(repeat, 1, 1, 1)
        return x + feat


def inject_inpaint_features(unet, features: Tensor) -> AddInpaintFeature:
    """Inject InpaintHead features into the UNet's first input block.

    Inserts a module into the first DownBlock that adds the pre-computed
    features to the Conv2d output, before the ResidualAccumulator saves
    the skip connection.

    Returns the injected module (for MultiDiffusion tile cropping).
    """
    # Navigate to first DownBlock
    # SDXLUNet chain: TimestepEncoder, DownBlocks, MiddleBlock, Residual, UpBlocks, OutputBlock
    down_blocks = None
    for module in unet:
        if module.__class__.__name__ == "DownBlocks":
            down_blocks = module
            break

    if down_blocks is None:
        msg = "Could not find DownBlocks in UNet"
        raise RuntimeError(msg)

    first_block = next(iter(down_blocks))

    # Insert after Conv2d (index 0), before ResidualAccumulator
    injector = AddInpaintFeature(features)
    injector.to(device=features.device, dtype=features.dtype)
    first_block.insert(1, injector)

    logger.info("Injected InpaintHead features into UNet first block")
    return injector

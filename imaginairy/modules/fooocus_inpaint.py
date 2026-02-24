"""Fooocus inpaint patch for SDXL.

Uses a tiny InpaintHead conv + LoRA patches to enable high-quality inpainting
with standard SDXL weights (no dedicated inpainting model needed).

References:
- https://github.com/Acly/comfyui-inpaint-nodes
- https://huggingface.co/lllyasviel/fooocus_inpaint
"""

import logging
from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image
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
    """Apply a weight delta to a parameter, cloning first to preserve cached model."""
    param.data = param.data.clone()
    param.data += alpha * delta.to(device=param.device, dtype=param.dtype)


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
    import numpy as np

    device = sd.unet.device
    dtype = sd.unet.dtype

    mask = mask_image.convert("L")
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_binary = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
    mask_binary = (mask_binary > 0.5).float()

    # Create masked PIL image (pixels zeroed in masked regions) and encode
    masked_arr = np.array(init_image.convert("RGB")).astype(np.float32)
    mask_arr = np.array(mask).astype(np.float32) / 255.0
    for c in range(3):
        masked_arr[:, :, c] *= 1 - mask_arr
    masked_pil = Image.fromarray(masked_arr.astype(np.uint8))

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


def inject_inpaint_features(unet, features: Tensor):
    """Inject InpaintHead features into the UNet's first input block.

    Inserts a module into the first DownBlock that adds the pre-computed
    features to the Conv2d output, before the ResidualAccumulator saves
    the skip connection.
    """
    import imaginairy.vendored.refiners.fluxion.layers as fl

    class AddInpaintFeature(fl.Module):
        def __init__(self, feature: Tensor):
            super().__init__()
            self.register_buffer("feature", feature)

        def forward(self, x: Tensor) -> Tensor:
            # Handle CFG batch doubling (UNet gets 2x batch for guidance)
            repeat = x.shape[0] // self.feature.shape[0]
            feat = self.feature.repeat(repeat, 1, 1, 1)
            return x + feat

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

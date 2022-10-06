from functools import lru_cache
from typing import Optional, Sequence

import cv2
import numpy as np
import PIL.Image
import torch
from torchvision import transforms

from imaginairy.img_log import log_img
from imaginairy.img_utils import pillow_fit_image_within
from imaginairy.vendored.clipseg import CLIPDensePredT

weights_url = "https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth"


@lru_cache()
def clip_mask_model():
    from imaginairy import PKG_ROOT  # noqa

    model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64, complex_trans_conv=True)
    model.eval()

    model.load_state_dict(
        torch.load(
            f"{PKG_ROOT}/vendored/clipseg/rd64-uni-refined.pth",
            map_location=torch.device("cpu"),
        ),
        strict=False,
    )
    return model


def get_img_mask(
    img: PIL.Image.Image,
    mask_description_statement: str,
    threshold: Optional[float] = None,
):
    from imaginairy.enhancers.bool_masker import MASK_PROMPT  # noqa

    parsed = MASK_PROMPT.parseString(mask_description_statement)
    parsed_mask = parsed[0][0]
    descriptions = list(parsed_mask.gather_text_descriptions())
    orig_size = img.size
    img = pillow_fit_image_within(img, max_height=352, max_width=352)
    mask_cache = get_img_masks(img, descriptions)
    mask = parsed_mask.apply_masks(mask_cache)
    log_img(mask, "combined mask")

    kernel = np.ones((3, 3), np.uint8)
    mask_g = mask.clone()

    # trial and error shows 0.5 threshold has the best "shape"
    if threshold is not None:
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
    log_img(mask, f"mask threshold {0.5}")

    mask_np = mask.to(torch.float32).cpu().numpy()
    smoother_strength = 2
    # grow the mask area to make sure we've masked the thing we care about
    for _ in range(smoother_strength):
        mask_np = cv2.dilate(mask_np, kernel)
    # todo: add an outer blur (not gaussian)
    mask = torch.from_numpy(mask_np)
    log_img(mask, "mask after closing (dilation then erosion)")

    mask_img = transforms.ToPILImage()(mask).resize(
        orig_size, resample=PIL.Image.Resampling.LANCZOS
    )
    mask_img_g = transforms.ToPILImage()(mask_g).resize(
        orig_size, resample=PIL.Image.Resampling.LANCZOS
    )
    return mask_img, mask_img_g


def get_img_masks(img, mask_descriptions: Sequence[str]):
    a, b = img.size
    orig_size = b, a
    log_img(img, "image for masking")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ]
    )
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        preds = clip_mask_model()(
            img.repeat(len(mask_descriptions), 1, 1, 1), mask_descriptions
        )[0]
    preds = transforms.Resize(orig_size)(preds)

    preds = [torch.sigmoid(p[0]) for p in preds]

    preds_dict = {}
    for p, desc in zip(preds, mask_descriptions):
        log_img(p, f"clip mask: {desc}")
        preds_dict[desc] = p

    return preds_dict

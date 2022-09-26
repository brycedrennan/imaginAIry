from functools import lru_cache
from typing import Optional, Sequence

import cv2
import numpy as np
import PIL.Image
import torch
from kornia.filters import median_blur
from torchvision import transforms

from imaginairy.img_log import log_img
from imaginairy.vendored.clipseg import CLIPDensePredT

weights_url = "https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth"


@lru_cache()
def clip_mask_model():
    from imaginairy import PKG_ROOT  # noqa

    model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
    model.eval()

    model.load_state_dict(
        torch.load(
            f"{PKG_ROOT}/vendored/clipseg/rd64-uni.pth",
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
    mask_cache = get_img_masks(img, descriptions)
    mask = parsed_mask.apply_masks(mask_cache)
    log_img(mask, "combined mask")

    # try to blur the square shaped artifacts somewhat
    mask = median_blur(mask.unsqueeze(dim=0).unsqueeze(dim=0), (11, 11)).squeeze()
    log_img(mask, "median blurred")

    kernel = np.ones((5, 5), np.uint8)
    mask_g = mask.clone()

    # trial and error shows 0.5 threshold has the best "shape"
    if threshold is not None:
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
    log_img(mask, f"mask threshold {0.5}")

    mask_np = mask.cpu().numpy()
    smoother_strength = 5
    # grow the mask area to make sure we've masked the thing we care about
    for _ in range(smoother_strength):
        mask_np = cv2.dilate(mask_np, kernel)
    # todo: add an outer blur (not gaussian)
    mask = torch.from_numpy(mask_np)
    log_img(mask, "mask after closing (dilation then erosion)")

    return transforms.ToPILImage()(mask), transforms.ToPILImage()(mask_g)


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

from functools import lru_cache

import torch
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


def get_img_mask(img, mask_description, negative_description=""):
    pos_descriptions = mask_description.split("|")
    pos_masks = get_img_masks(img, pos_descriptions)
    pos_mask = pos_masks[0]
    for pred in pos_masks:
        pos_mask = torch.maximum(pos_mask, pred)

    log_img(pos_mask, "pos mask")

    if negative_description:
        neg_descriptions = negative_description.split("|")
        neg_masks = get_img_masks(img, neg_descriptions)
        neg_mask = neg_masks[0]
        for pred in neg_masks:
            neg_mask = torch.maximum(neg_mask, pred)
        neg_mask = (neg_mask * 3).clip(0, 1)
        log_img(neg_mask, "neg_mask")
        pos_mask = torch.minimum(pos_mask, 1 - neg_mask)
    _min = pos_mask.min()
    _max = pos_mask.max()
    _range = _max - _min
    pos_mask = (pos_mask > (_min + (_range * 0.35))).float()

    return transforms.ToPILImage()(pos_mask)


def get_img_masks(img, mask_descriptions):
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
    preds = transforms.GaussianBlur(kernel_size=9)(preds)

    preds = [torch.sigmoid(p[0]) for p in preds]

    bw_preds = []
    for p, desc in zip(preds, mask_descriptions):
        log_img(p, f"clip mask: {desc}")
        # bw_preds.append(pred_transform(p))

        bw_preds.append(p)

    return bw_preds

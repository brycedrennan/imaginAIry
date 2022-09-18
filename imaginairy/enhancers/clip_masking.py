from functools import lru_cache

import torch
from torchvision import transforms

from imaginairy.img_log import log_img
from imaginairy.vendored.clipseg import CLIPDensePredT

weights_url = "https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth"


@lru_cache()
def clip_mask_model():
    from imaginairy import PKG_ROOT

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


def get_img_mask(img, mask_description):
    return get_img_masks(img, [mask_description])[0]


def get_img_masks(img, mask_descriptions):
    a, b = img.size
    orig_size = b, a
    log_img(img, "image for masking")
    # orig_shape = tuple(img.shape)[1:]
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
    bw_preds = []
    for p in preds:
        log_img(p, f"clip mask for {mask_descriptions}")
        # bw_preds.append(pred_transform(p))
        _min = p.min()
        _max = p.max()
        _range = _max - _min
        p = (p > (_min + (_range * 0.5))).float()
        bw_preds.append(transforms.ToPILImage()(p))

    return bw_preds

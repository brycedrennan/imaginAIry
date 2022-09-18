from functools import lru_cache

import torch
from torchvision import transforms

from imaginairy import PKG_ROOT
from imaginairy.vendored.clipseg import CLIPDensePredT

weights_url = "https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth"


@lru_cache()
def clip_mask_model():
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()

    model.load_state_dict(
        torch.load(
            f'{PKG_ROOT}/vendored/clipseg/rd64-uni.pth',
            map_location=torch.device("cpu")),
        strict=False
    )
    return model


def get_img_mask(img, mask_descriptions):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        preds = clip_mask_model()(img.repeat(len(mask_descriptions), 1, 1, 1), mask_descriptions)[0]

    return preds

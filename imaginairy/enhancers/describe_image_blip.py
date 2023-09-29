import os
import os.path
from functools import lru_cache

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from imaginairy.model_manager import get_cached_url_path
from imaginairy.utils import get_device
from imaginairy.vendored.blip.blip import BLIP_Decoder, load_checkpoint

device = get_device()
if "mps" in device:
    device = "cpu"

BLIP_EVAL_SIZE = 384


@lru_cache
def blip_model():
    from imaginairy.paths import PKG_ROOT

    config_path = os.path.join(
        PKG_ROOT, "vendored", "blip", "configs", "med_config.json"
    )
    url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth"

    model = BLIP_Decoder(image_size=BLIP_EVAL_SIZE, vit="base", med_config=config_path)
    cached_url_path = get_cached_url_path(url)
    model, msg = load_checkpoint(model, cached_url_path)
    model.eval()
    model = model.to(device)
    return model


def generate_caption(image, min_length=30):
    """Given an image, return a caption."""
    image = image.convert("RGB")
    gpu_image = (
        transforms.Compose(
            [
                transforms.Resize(
                    (BLIP_EVAL_SIZE, BLIP_EVAL_SIZE),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        caption = blip_model().generate(
            gpu_image, sample=True, num_beams=3, max_length=80, min_length=min_length
        )
    return caption[0]
